"""
Rollout module for GRPO training.

Orchestrates the generation phase: sample responses, compute rewards, 
and package everything into Experience for training.
"""
from __future__ import annotations

from typing import Callable, Any, Optional
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from teacher_torch.sampling import SamplingOutput, sample_responses_hf
from teacher_torch.experience import ExperienceBuffer
from teacher_torch.loss import compute_group_advantages, sequences_log_probs
from teacher_torch.utils import setup_logger_with_file_line

logger = setup_logger_with_file_line(__name__)


@dataclass
class RolloutConfig:
    """Configuration for rollout behavior.
    
    Attributes:
        group_size: Number of samples per prompt for group-relative advantages.
        seed: Random seed for reproducibility.
        normalize_advantages: Whether to normalize advantages (mean=0, std=1).
        use_rloo_baseline: Use RLOO baseline (leave-one-out) for variance reduction.
    """
    group_size: int = 4
    seed: int = 42
    normalize_advantages: bool = True
    use_rloo_baseline: bool = False

@dataclass 
class RolloutMetric:
    """Metrics collected during a rollout.
    
    Attributes:
        num_samples: Total number of samples generated.
        group_size: Number of samples per prompt group.
        mean_reward: Mean reward across all samples (None if no reward_fn provided).
        std_reward: Standard deviation of rewards (None if no reward_fn provided).
        mean_entropy: Mean entropy of the policy distribution.
        mean_completion_len: Mean length of generated completions.
        extra_metrics: Additional metrics from reward_fn (None if no reward_fn provided).
    """
    num_samples: int
    group_size: int
    mean_entropy: float
    mean_completion_len: float
    mean_reward: Optional[float] = None
    std_reward: Optional[float] = None
    extra_metrics: Optional[dict[str, Any]] = None 


# Type alias for reward functions
RewardFn = Callable[[SamplingOutput], dict[str, torch.FloatTensor|Any]]


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    generation_config: GenerationConfig,
    config: RolloutConfig,
    role: str,
    solved_threshold: float = 1.0,
    reward_fn: Optional[RewardFn] = None,
    logits_minibatch_size: int = 128,
) -> tuple[ExperienceBuffer, RolloutMetric]:
    """
    Perform a single rollout: sample responses and compute experiences.
    
    This is the main entry point for the generation phase of GRPO. It:
    1. Samples multiple responses per prompt (group_size)
    2. Computes log probabilities for importance sampling
    3. Calls reward_fn to score each response
    4. Computes group-relative advantages
    5. Packages everything into Experience for training
    
    Args:
        model: The policy model to sample from.
        tokenizer: Tokenizer for the model.
        prompts: List of prompt strings.
        reward_fn: Callable that takes SamplingOutput and returns dict with:
            - "rewards": Tensor of shape (num_samples,) with scalar rewards
            - Any additional keys for logging (e.g., "accuracy", "format_ok")
        generation_config: HuggingFace generation config.
            Note: num_return_sequences should equal config.group_size.
        config: RolloutConfig with hyperparameters.
        role: Either "teacher" or "student" for logging.
        solved_threshold: Threshold for considering a prompt group solved.
        reward_fn: Callable that takes SamplingOutput and returns dict with:
            - "rewards": Tensor of shape (num_samples,) with scalar rewards
            - Any additional keys for logging (e.g., "accuracy", "format_ok")
    
    Returns:
        Tuple of:
            - ExperienceBuffer: Container with all data needed for GRPO training.
            - RolloutMetric: Metrics from reward_fn plus rollout stats.
    
    Example:
        >>> def my_reward_fn(sampling_output: SamplingOutput) -> dict[str, Any]:
        ...     # Your reward logic here
        ...     rewards = torch.tensor([score(c) for c in sampling_output.completions])
        ...     return {"rewards": rewards, "accuracy": rewards.mean()}
        >>> 
        >>> exp, metrics = rollout(
        ...     model, tokenizer, prompts=["Solve: 2+2="],
        ...     reward_fn=my_reward_fn,
        ...     generation_config=gen_config,
        ...     config=RolloutConfig(group_size=4),
        ...     role="student",
        ... )
    """
    # Sanity check: generation config must match group_size
    assert generation_config.num_return_sequences == config.group_size, (
        f"generation_config.num_return_sequences ({generation_config.num_return_sequences}) "
        f"must equal config.group_size ({config.group_size})"
    )
    batch_size = len(prompts)
    num_samples = batch_size * config.group_size

    logger.info(f"# Unique Input Prompts: {batch_size}")
    logger.info(f"# Total Samples: {num_samples}")
    logger.info(f"# Group Size: {config.group_size}")
    logger.info(f"# First Input Prompt:\n{prompts[0]}")

    # === Step 1: Sample responses ===
    sampling_output = sample_responses_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=generation_config,
        seed=config.seed,
    )
    logger.info(f"\033[34mFirst sampling output:\n======\n{sampling_output.completions[0]}\033[0m")
    logger.info(f"Sampling Statistics:\n======\n{str(sampling_output)}")
    
    # === Step 2: Compute old log probs (behavior policy) ===
    # These are needed for importance sampling in PPO/GRPO
    action_log_probs, entropy = sequences_log_probs(
        model=model,
        sequence_ids=sampling_output.sequence_ids,
        sequence_attention_mask=sampling_output.sequence_attention_mask,
        return_entropy=True,
        logits_minibatch_size=logits_minibatch_size,
    )
    
    rewards, advantages, learnability = None, None, None
    if reward_fn is not None:
        # === Step 3: Compute rewards ===
        reward_output = reward_fn(sampling_output)
        rewards:torch.FloatTensor = reward_output["rewards"]  # (num_samples,)
        
        assert rewards.shape == (num_samples,), (
            f"reward_fn must return rewards of shape ({num_samples},), got {rewards.shape}"
        )
        
        # === Step 4: Compute group-relative advantages ===
        # Reshape rewards into groups: (batch_size, group_size)
        rewards_grouped = rewards.view(batch_size, config.group_size)
        
        advantages_grouped = compute_group_advantages(
            returns=rewards_grouped,
            normalize=config.normalize_advantages,
            use_rloo_scalar=config.use_rloo_baseline,
        )
        
        # Flatten back: (batch_size * group_size,)
        advantages = advantages_grouped.view(-1)
        
        # === Step 5: Package into Experience ===        
        # learnability: compute based on solved status for each prompt group
        # Formula: learnability = solved * (1 - solved_mean_in_group)
        # Shape: (num_samples,) - one per sample
        solved_mask = (rewards >= solved_threshold).bool()  # (num_samples,)
        solved_float = solved_mask.float()  # (num_samples,)
        
        # Reshape to groups: (batch_size, group_size)
        solved_grouped = solved_float.view(batch_size, config.group_size)
        # Compute mean solved rate per group: (batch_size,)
        solved_mean_per_group = solved_grouped.mean(dim=1)  # (batch_size,)
        # Expand back to per-sample: (num_samples,)
        solved_mean_per_sample = solved_mean_per_group.repeat_interleave(config.group_size)
        
        # Compute learnability: solved[i] * (1 - mean(solved in group))
        learnability = solved_float * (1.0 - solved_mean_per_sample)

        # === Collect metrics ===
        extra_metrics = {k: v for k, v in reward_output.items() if k != "rewards"}
    
        metrics = RolloutMetric(
            num_samples=num_samples,
            group_size=config.group_size,
            mean_entropy=entropy.mean().item() if entropy is not None else 0.0,
            mean_completion_len=sampling_output.action_mask.sum(dim=1).float().mean().item(),
            mean_reward=rewards.mean().item(),
            std_reward=rewards.std().item(),
            extra_metrics=extra_metrics,
        )
    else:
        # No reward_fn provided - reward-related metrics are None
        metrics = RolloutMetric(
            num_samples=num_samples,
            group_size=config.group_size,
            mean_entropy=entropy.mean().item() if entropy is not None else 0.0,
            mean_completion_len=sampling_output.action_mask.sum(dim=1).float().mean().item(),
            mean_reward=None,
            std_reward=None,
            extra_metrics=None,
        )
    
    # Prepare prompts and completions for ExperienceBuffer
    # prompts should be unique prompts (one per group), length = batch_size (number of groups)
    # completions should be flat list of all completions, length = num_samples (total samples)
    experience = ExperienceBuffer.from_sampling_output(
        role=role,
        sampling_output=sampling_output,
        pad_token_id=tokenizer.pad_token_id,
        group_size=config.group_size,
        prompts=prompts,
        completions=sampling_output.completions,
        action_log_probs=action_log_probs,
        rewards=rewards,
        advantages=advantages,
        learnability=learnability,
        solved_threshold=solved_threshold,
    )
    
    return experience, metrics
