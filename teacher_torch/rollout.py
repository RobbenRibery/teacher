"""
Rollout module for GRPO training.

Orchestrates the generation phase: sample responses, compute rewards, 
and package everything into Experience for training.
"""
from __future__ import annotations

from typing import Callable, Any
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from teacher_torch.sampling import SamplingOutput, sample_responses_hf
from teacher_torch.experience import Experience
from teacher_torch.loss import compute_group_advantages, sequences_log_probs


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


# Type alias for reward functions
RewardFn = Callable[[SamplingOutput], dict[str, Any]]


@torch.no_grad()
def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    reward_fn: RewardFn,
    generation_config: GenerationConfig,
    config: RolloutConfig,
    role: str = "student",
) -> tuple[Experience, dict[str, Any]]:
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
    
    Returns:
        Tuple of:
            - Experience: Container with all data needed for GRPO training.
            - dict: Metrics from reward_fn plus rollout stats.
    
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
        ... )
    """
    # Sanity check: generation config must match group_size
    assert generation_config.num_return_sequences == config.group_size, (
        f"generation_config.num_return_sequences ({generation_config.num_return_sequences}) "
        f"must equal config.group_size ({config.group_size})"
    )
    
    batch_size = len(prompts)
    num_samples = batch_size * config.group_size
    
    # === Step 1: Sample responses ===
    sampling_output = sample_responses_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=generation_config,
        seed=config.seed,
    )
    
    # === Step 2: Compute old log probs (behavior policy) ===
    # These are needed for importance sampling in PPO/GRPO
    attention_mask = (sampling_output.sequence_ids != tokenizer.pad_token_id).long()
    
    old_log_probs, entropy = sequences_log_probs(
        model=model,
        sequence_ids=sampling_output.sequence_ids,
        attention_mask=attention_mask,
        return_entropy=True,
    )
    
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
    experience = Experience.from_sampling_output(
        sampling_output=sampling_output,
        pad_token_id=tokenizer.pad_token_id,
        old_log_probs=old_log_probs.cpu(),
        rewards=rewards.cpu(),
        advantages=advantages.cpu(),
        group_size=config.group_size,
    )
    
    # === Collect metrics ===
    metrics = {
        "num_samples": num_samples,
        "group_size": config.group_size,
        "mean_reward": rewards.mean().item(),
        "std_reward": rewards.std().item(),
        "mean_entropy": entropy.mean().item() if entropy is not None else 0.0,
        "mean_completion_len": sampling_output.action_mask.sum(dim=1).float().mean().item(),
        **{k: v for k, v in reward_output.items() if k != "rewards"},  # pass through extra metrics
    }
    
    return experience, metrics
