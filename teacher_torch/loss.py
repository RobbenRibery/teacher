"""
GRPO (Group Relative Policy Optimization) Loss Implementation.

This module implements the core loss functions for GRPO training:
- PPO-style clipped surrogate objective
- KL regularization against a reference policy
- Group-relative advantage normalization

Functions are compiled with torch.compile(dynamic=True) for acceleration
with variable sequence lengths.

References:
    - Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    - DeepSeek-AI, "DeepSeek-R1" (2024) - GRPO formulation
"""

from __future__ import annotations
from typing import Optional

from transformers import PreTrainedModel
import torch

from teacher_torch.utils import release_memory
from teacher_torch.utils import setup_logger_with_file_line

logger = setup_logger_with_file_line(__name__)


# ============================================================================
# Core loss functions (compiled via decorator)
# ============================================================================
@torch.compile(dynamic=True)
def approx_kl_divergence(
    log_probs: torch.FloatTensor,
    log_probs_ref: torch.FloatTensor,
    action_mask: Optional[torch.BoolTensor] = None,
) -> torch.FloatTensor:
    """
    Monte-Carlo approximation of KL divergence, kl3 estimator, see: http://joschu.net/blog/kl-approx.html
    Args:
        log_probs (torch.FloatTensor): The log probabilities of the current model.
        log_probs_ref (torch.FloatTensor): The log probabilities of the reference model.
        action_mask (Optional[torch.BoolTensor]): The action mask.

    Returns:
        torch.FloatTensor: The KL divergence.
    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


@torch.compile(dynamic=True)
def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dim: int = None,
    use_fixed_response_length: bool = False,
    max_new_tokens: int = 2048,
) -> torch.Tensor:
    """
    Compute the mean of the tensor over the specified dimension.
    If mask is None, return the mean of the whole tensor.
    If dim is None, return the mean of the whole tensor.

    Args:
        tensor (torch.Tensor): The tensor to compute the mean of. (batch_size, seq_len)
        mask (Optional[torch.Tensor]): The mask to compute the mean of. (batch_size, seq_len)
        dim (Optional[int]): The dimension to compute the mean of. default to None

    Returns:
        torch.Tensor: The mean of the tensor.
    """
    if mask is None:
        return tensor.mean(axis=dim)
    # when dim == None, return the mean of the whole tensor
    if use_fixed_response_length:
        return (tensor * mask).sum(axis=dim) / max_new_tokens
    else:
        return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


@torch.compile(dynamic=True)
def compute_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute PPO clipped surrogate policy loss.

    Implements the clipped surrogate objective from PPO:
    L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]

    where r_t = π(a|s) / π_old(a|s) is the importance sampling ratio.

    Args:
        log_probs: Current policy log probabilities.
            Shape: (batch_size, seq_len)
        old_log_probs: Behavior policy log probabilities (from rollout).
            Shape: (batch_size, seq_len)
        advantages: Per-sequence advantages (will be broadcast to tokens).
            Shape: (batch_size,)
        action_mask: Binary mask for action tokens.
            Shape: (batch_size, seq_len)
        clip_eps: Clipping epsilon for importance ratio. Default: 0.2

    Returns:
        Scalar policy loss (to be minimized).
    """
    # Importance sampling ratio: π(a|s) / π_old(a|s)
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Broadcast advantages: (B,) -> (B, 1) for token-level multiplication
    adv_broadcast = advantages.unsqueeze(1)

    # Clipped surrogate objective
    surr1 = ratio * adv_broadcast
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_broadcast

    # Take min and negate (we minimize negative expected reward)
    policy_loss_per_token = -torch.min(surr1, surr2)

    return masked_mean(policy_loss_per_token, action_mask, dim=None)


@torch.compile(dynamic=True)
def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute KL penalty term for regularization.

    Computes the mean KL divergence across all action tokens,
    used to penalize deviation from the reference policy.

    Args:
        log_probs: Current policy log probabilities.
            Shape: (batch_size, seq_len)
        ref_log_probs: Reference policy log probabilities.
            Shape: (batch_size, seq_len)
        action_mask: Binary mask for action tokens.
            Shape: (batch_size, seq_len)

    Returns:
        Scalar mean KL divergence.
    """
    per_token_kl = log_probs - ref_log_probs
    return masked_mean(per_token_kl, action_mask, dim=None)


@torch.compile(dynamic=True)
def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    ref_log_probs: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute GRPO loss with PPO clipping and KL regularization.

    The total loss combines two components:
    1. PPO clipped surrogate objective for stable policy updates
    2. KL penalty to prevent drift from reference policy

    Loss = L^CLIP + β * KL(π || π_ref)

    Args:
        log_probs: Current policy log probabilities.
            Shape: (batch_size, seq_len)
        old_log_probs: Behavior policy log probabilities from rollout.
            Shape: (batch_size, seq_len)
        ref_log_probs: Optional[Reference policy log probabilities (frozen model).
            Shape: (batch_size, seq_len)]
        advantages: Group-relative advantages for each sequence.
            Shape: (batch_size,)
            Should be normalized within groups (mean=0, std=1).
        action_mask: Binary mask for action tokens (1=action, 0=prompt/pad).
            Shape: (batch_size, seq_len)
        clip_eps: PPO clipping epsilon. Default: 0.2
            Constrains importance ratio to [1-ε, 1+ε].
        kl_beta: KL penalty coefficient. Default: 0.04
            Higher values enforce stricter adherence to reference.

    Returns:
        Tuple of:
            - total_loss: Scalar combined loss (policy + KL).
            - mean_kl: Scalar mean KL divergence for monitoring.
            - policy_loss: Scalar policy loss for monitoring.

    Example:
        >>> loss, kl, pg_loss = grpo_loss(
        ...     log_probs, old_log_probs, ref_log_probs,
        ...     advantages, action_mask,
        ...     clip_eps=0.2, kl_beta=0.04
        ... )
        >>> loss.backward()
    """
    # Policy loss with PPO clipping
    policy_loss = compute_policy_loss(
        log_probs, old_log_probs, advantages, action_mask, clip_eps
    )

    # KL penalty
    if ref_log_probs is not None:
        mean_kl = compute_kl_penalty(log_probs, ref_log_probs, action_mask)
        total_loss = policy_loss + kl_beta * mean_kl
    else:
        mean_kl = torch.tensor(0.0, device=log_probs.device)
        total_loss = policy_loss

    return total_loss, mean_kl, policy_loss


@torch.compile(dynamic=True)
def compute_group_advantages(
    returns: torch.FloatTensor,
    eps: float = 1e-8,
    normalize: bool = True,
    use_rloo_scalar: bool = False,
) -> torch.FloatTensor:
    """
    Normalizes the advantages of a group of returns.

    Args:
        returns (torch.Tensor): The returns to normalize. (num_samples, num_return_sequences)
        eps (float): The epsilon value to add to the standard deviation to prevent division by zero.

    Returns:
        torch.Tensor: The normalized advantages. (num_samples * group_size,)
    """
    centered_returns = returns - returns.mean(dim=1, keepdim=True)

    if normalize:
        centered_returns = centered_returns / (
            centered_returns.std(dim=1, keepdim=True) + eps
        )

    if use_rloo_scalar:
        # unbiased scalar G / (G-1)
        centered_returns = centered_returns * (
            returns.shape[1] / (returns.shape[1] - 1)
        )

    return centered_returns.view(-1)


@torch.compile(dynamic=True)
def _sequence_log_probs_from_logits(
    logits: torch.FloatTensor,
    output_ids: torch.IntTensor,
    return_entropy: bool = False,
) -> tuple[torch.FloatTensor, torch.FloatTensor | None]:
    """
    Computes the log probabilities of the output ids from the logits.

    Optimized version using log_softmax for better performance.

    Args:
        logits (torch.FloatTensor): The logits of the model. (num_samples * num_rollouts, seq_len, vocab_size)
        output_ids (torch.IntTensor): The output ids to compute the log probabilities for. (num_samples * num_rollouts, seq_len)
        return_entropy (bool): Whether to return the entropy of the distribution.
            Default: False.
    Returns:
        torch.FloatTensor: The log probabilities of the output ids. (num_samples * num_rollouts, seq_len)
        torch.FloatTensor | None: The entropy of the distribution. None if return_entropy is False.
    """

    # Compute log_softmax once (more efficient than manual logsumexp)
    log_probs_all = torch.nn.functional.log_softmax(
        logits, dim=-1
    )  # (num_samples * num_rollouts, seq_len, vocab_size)

    # Gather the log probabilities corresponding to the output_ids
    log_probs = log_probs_all.gather(dim=-1, index=output_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (num_samples * num_rollouts, seq_len)

    if return_entropy:
        # Optimized entropy computation: H = -sum(p * log(p))
        # where p = exp(log_softmax), so H = -sum(exp(log_softmax) * log_softmax)
        # This reuses log_probs_all and avoids separate softmax computation
        probs = torch.exp(
            log_probs_all
        )  # (num_samples * num_rollouts, seq_len, vocab_size)
        entropy = -torch.sum(
            probs * log_probs_all, dim=-1
        )  # (num_samples * num_rollouts, seq_len)
        return log_probs, entropy
    else:
        return log_probs, None

@torch.compile(dynamic=True)
def sequences_log_probs(
    model: PreTrainedModel,
    sequence_ids: torch.Tensor,
    sequence_attention_mask: torch.Tensor,
    return_entropy: bool = True,
    logits_minibatch_size: int = 128,
    ref_model: PreTrainedModel | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Computes the log probabilities of the output ids from the logits for both
    the current model and optionally a reference model in parallel using CUDA streams.

    Args:
        model (PreTrainedModel): The model to compute the log probabilities for.
        sequence_ids (torch.Tensor): The sequence ids to compute the log probabilities for.
            (num_samples * group_size, seq_len)
        sequence_attention_mask (torch.Tensor): The attention mask to compute the log probabilities for.
            (num_samples * group_size, seq_len)
        return_entropy (bool): Whether to return entropy for the current model. Default: True.
        logits_minibatch_size (int): Batch size for processing logits. Default: 128.
        ref_model (PreTrainedModel | None): The reference model to compute log probs for.
            If None, ref_log_probs will be None in the output.
        log_probs_ref (torch.Tensor | None): The log probabilities from the reference model.
            If None, ref_log_probs will be computed from the reference model.

    Returns:
        tuple containing:
            - log_probs: The log probabilities from the current model. (num_samples * group_size, seq_len-1)
            - entropy: The entropy from the current model (None if return_entropy=False).
            - ref_log_probs: The log probabilities from the reference model (None if ref_model is None).
    """
    token_logprobs: list[torch.Tensor] = []
    ref_token_logprobs: list[torch.Tensor] = []
    if return_entropy:
        token_entropy: list[torch.Tensor] = []

    for i in range(0, sequence_ids.shape[0], logits_minibatch_size):
        mini_ids = sequence_ids[i : i + logits_minibatch_size]
        mini_mask = sequence_attention_mask[i : i + logits_minibatch_size]

        # Sequential execution is safer and often just as fast due to GPU saturation
        mini_output = model(
            input_ids=mini_ids, attention_mask=mini_mask, use_cache=False
        )
        mini_logits: torch.FloatTensor = mini_output["logits"]
        del mini_output

        if ref_model is not None:
            with torch.inference_mode():
                ref_mini_output = ref_model(
                    input_ids=mini_ids, attention_mask=mini_mask, use_cache=False
                )
                ref_mini_logits: torch.FloatTensor = ref_mini_output["logits"]
                del ref_mini_output
        else:
            ref_mini_logits = None

        logger.info(f"Mini logits shape: {mini_logits.shape}")
        del mini_mask

        # Compute log probs for current model
        # logits: [batch_size * num_rollouts, seq_len, vocab_size]
        log_probs, entropy = _sequence_log_probs_from_logits(
            logits=mini_logits[:, :-1],#.clone(),
            output_ids=mini_ids[:, 1:],  # right shift 1 block to get the actual output ids
            return_entropy=return_entropy,
        )
        del mini_logits

        ref_log_probs = None
        # Compute log probs for reference model (no entropy needed)
        if ref_mini_logits is not None:
            ref_log_probs, _ = _sequence_log_probs_from_logits(
                logits=ref_mini_logits[:, :-1],#.clone(),
                output_ids=mini_ids[:, 1:],
                return_entropy=False,
            )
            del ref_mini_logits
        
        release_memory(vars=[mini_ids])

        token_logprobs.append(log_probs)
        if return_entropy:
            token_entropy.append(entropy)
        
        if ref_log_probs is not None:
            ref_token_logprobs.append(ref_log_probs)
    
    reconstructed_logprobs = torch.cat(token_logprobs, dim=0)
    reconstructed_entropy = torch.cat(token_entropy, dim=0) if return_entropy else None
    reconstructed_ref_logprobs = torch.cat(ref_token_logprobs, dim=0) if ref_model is not None else None
    
    logger.info(f"Reconstructed logprobs shape: {reconstructed_logprobs.shape}")
    logger.info(
        f"Reconstructed entropy shape: {reconstructed_entropy.shape if reconstructed_entropy is not None else None}"
    )
    if reconstructed_ref_logprobs is not None:
        logger.info(f"Reconstructed ref_logprobs shape: {reconstructed_ref_logprobs.shape}")
    
    return reconstructed_logprobs, reconstructed_entropy, reconstructed_ref_logprobs
