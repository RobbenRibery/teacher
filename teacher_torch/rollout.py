"""
Rollout module for GRPO training.

Orchestrates the generation phase: sample responses, compute rewards,
and package everything into Experience for training.
"""

from __future__ import annotations

from typing import Callable, Optional, Type, Union
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig
from environment.reward import CurriculumReward, AnswerReward

from teacher_torch.sampling import sample_responses_hf
from teacher_torch.experience import TeacherExperience, StudentExperienceBuffer
from teacher_torch.loss import compute_group_advantages, sequences_log_probs
from teacher_torch.utils import setup_logger_with_file_line

logger = setup_logger_with_file_line(__name__)


@dataclass
class RolloutConfig:
    """Configuration for rollout behavior.
    Attributes:
        seed: Random seed for reproducibility.
        normalize_advantages: Whether to normalize advantages (mean=0, std=1).
        use_rloo_baseline: Use RLOO baseline (leave-one-out) for variance reduction.
    """
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
    mean_solved_rate: Optional[float] = None
    mean_learnability: Optional[float] = None
    mean_advantages: Optional[float] = None
    # extra_metrics: Optional[dict[str, Any]] = None


# Type alias for reward functions
RewardFn = Callable[[str], Type[CurriculumReward | AnswerReward]]


@torch.inference_mode()
def rollout(
    model: PreTrainedModel,
    reward_fn: RewardFn,
    tokenizer: PreTrainedTokenizer,
    prompts: list[str],
    generation_config: GenerationConfig,
    rollout_config: RolloutConfig,
    role: str,
    solved_threshold: float = 1.0,
    logits_minibatch_size: int = 128,
    only_check_format: bool = False,
    source_curriculums: Optional[list[str]] = None,
    ref_model: Optional[PreTrainedModel] = None,
    output_dir: Optional[Path] = None,
) -> tuple[Union[TeacherExperience, StudentExperienceBuffer], RolloutMetric]:
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
        reward_fn: Reward function to score the completions.
        tokenizer: Tokenizer for the model.
        prompts: List of prompt strings.
        generation_config: HuggingFace generation config.
            Note: num_return_sequences should equal config.group_size.
        rollout_config: RolloutConfig with hyperparameters.
        role: Either "teacher" or "student" for logging.
        solved_threshold: Threshold for considering a prompt group solved.
        reward_fn: Callable that takes completion string and returns Reward object.
        logits_minibatch_size: Batch size for computing log probabilities.
        only_check_format: Whether to only check format without computing advantages.
        source_curriculums: For student role, list of curriculum strings that seeded
            each prompt. Required when role=="student" to create StudentExperienceBuffer.
        ref_model: The reference model to use for importance sampling.
        output_dir: Directory to save the rollout output.
    Returns:
        Tuple of:
            - TeacherExperience or StudentExperienceBuffer: Container with all data
              needed for GRPO training. Returns StudentExperienceBuffer when
              role=="student" and source_curriculums is provided, otherwise returns
              TeacherExperience.
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
        ...     rollout_config=RolloutConfig(),
        ...     role="student",
        ... )
    """
    if not reward_fn:
        raise ValueError("Reward function is required")
    batch_size = len(prompts)  # -> number of unique questions/inputs
    group_size = generation_config.num_return_sequences
    num_samples = batch_size * group_size  # -> total number of samples

    logger.info(f"# Unique Input Prompts: {batch_size}")
    logger.info(f"# Total Samples: {num_samples}")
    logger.info(f"# Group Size: {group_size}")
    logger.info(f"# First Input Prompt:\n{prompts[0]}")

    # === Step 1: Sample responses ===
    sampling_output = sample_responses_hf(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        generation_config=generation_config,
        seed=rollout_config.seed,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, completion in enumerate(sampling_output.completions):
            with open(output_dir / f"completion_{i}.txt", "w") as f:
                f.write(completion)
            logger.info(f"Saved completion {i} to {output_dir / f'completion_{i}.txt'}")

    logger.info(f"Sampling Statistics:\n======\n{str(sampling_output)}")

    # === Step 2: Compute old log probs (behavior policy) ===
    # These are needed for importance sampling in PPO/GRPO
    action_log_probs, entropy, ref_log_probs = sequences_log_probs(
        model=model,
        ref_model=ref_model,
        sequence_ids=sampling_output.sequence_ids,
        sequence_attention_mask=sampling_output.sequence_attention_mask,
        return_entropy=True,
        logits_minibatch_size=logits_minibatch_size,
    )

    rewards, advantages, learnability = None, None, None
    reward_outputs: list[Type[CurriculumReward | AnswerReward]] = [
        reward_fn(completion) for completion in sampling_output.completions
    ]
    for i, reward_output in enumerate[type[CurriculumReward] | type[AnswerReward]](reward_outputs):
        logger.info(f"\033[35mReward output {i}: \n======\n{str(reward_output)}\033[0m")

    rewards: list[float] = [reward.reward for reward in reward_outputs]
    if any(reward is None for reward in reward_outputs):
        raise ValueError("reward_fn must return a Reward object for each completion")

    rewards: torch.FloatTensor = torch.tensor(
        rewards, device=model.device
    )
    assert rewards.shape == (num_samples,), (
        f"reward_fn must return rewards of shape ({num_samples},), got {rewards.shape}"
    )

    advantages, learnability = None, None
    mean_solved_rate, mean_learnability, mean_advantages = None, None, None
    if not only_check_format:
        # Compute advantages
        # advantages: (num_samples,)
        advantages = compute_group_advantages(
            returns=rewards.view(batch_size, group_size),
            normalize=rollout_config.normalize_advantages,
            use_rloo_scalar=rollout_config.use_rloo_baseline,
        )
        # Compute solved rate
        solved_mask = (rewards >= solved_threshold).bool()  # (num_samples,)
        solved_grouped = solved_mask.view(
            batch_size, group_size
        )  # (batch_size, group_size)
        solved_mean_per_group = solved_grouped.mean(dim=1)  # (batch_size,)

        # Compute learnability
        learnability: torch.FloatTensor = solved_mean_per_group * (
            1.0 - solved_mean_per_group
        )  # (batch_size,)

        # Compute mean solved rate
        mean_solved_rate = solved_mask.float().mean().item()
        mean_learnability = learnability.float().mean().item()
        mean_advantages = advantages.mean().item()

    # === Collect metrics ===
    metrics = RolloutMetric(
        num_samples=num_samples,
        group_size=group_size,
        mean_entropy=entropy.mean().item() if entropy is not None else 0.0,
        mean_completion_len=sampling_output.action_mask.sum(dim=1)
        .float()
        .mean()
        .item(),
        mean_reward=rewards.mean().item(),
        std_reward=rewards.std().item(),
        mean_solved_rate=mean_solved_rate,
        mean_learnability=mean_learnability,
        mean_advantages=mean_advantages,
    )

    # Create appropriate experience based on role
    if role == "student" and source_curriculums is not None:
        # Compute solved_mask for student experiences
        solved_mask = (rewards >= solved_threshold).bool() if rewards is not None else None
        
        experience = StudentExperienceBuffer.from_sampling_output(
            sampling_output=sampling_output,
            pad_token_id=tokenizer.pad_token_id,
            action_log_probs=action_log_probs,
            source_curriculums=source_curriculums,
            action_log_probs_ref=ref_log_probs,
            rewards=rewards,
            advantages=advantages,
            solved_mask=solved_mask,
            learnability=learnability,
        )
    else:
        # Teacher experience (single experience, not a buffer)
        experience = TeacherExperience.from_sampling_output(
            sampling_output=sampling_output,
            pad_token_id=tokenizer.pad_token_id,
            action_log_probs=action_log_probs,
            action_log_probs_ref=ref_log_probs,
            rewards=rewards,
            only_check_format=only_check_format,
            curriculums=[reward_output.extracted_curriculum for reward_output in reward_outputs],
        )

    return experience, metrics
