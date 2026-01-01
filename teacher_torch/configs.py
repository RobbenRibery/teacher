from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for generation."""

    max_new_tokens: int = 512
    temperature: float = 0.8
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.1
    seed: int = 256
    do_sample: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert generation config to dictionary."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }


@dataclass
class TeacherGenerationConfig(GenerationConfig):
    """Configuration for GRPO generation."""

    teacher_group_size: int = 4
    max_new_tokens: int = 4096
    stop_strings: Optional[list[str]] = None
    seed: Optional[int] = 512

    def to_dict(self) -> dict[str, Any]:
        """Convert teacher generation config to dictionary."""
        org = super().to_dict()
        org["num_return_sequences"] = self.teacher_group_size
        if self.stop_strings is not None:
            org["stop_strings"] = self.stop_strings
        return org


@dataclass
class StudentGenerationConfig(GenerationConfig):
    """Configuration for GRPO generation."""

    student_group_size: int = 16
    max_new_tokens: int = 4096
    stop_strings: Optional[list[str]] = None
    seed: Optional[int] = 1024
    temperature: float = 1.0
    top_k: int = 100
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    do_sample: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert student generation config to dictionary."""
        org = super().to_dict()
        org["num_return_sequences"] = self.student_group_size
        if self.stop_strings is not None:
            org["stop_strings"] = self.stop_strings
        return org


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # generation configuration (must come first - no defaults)
    teacher_generation_config: TeacherGenerationConfig
    student_generation_config: StudentGenerationConfig

    # model name or path
    mode_name_or_path: str = "Qwen/Qwen3-1.7B"
    minimal_legit_curriculum_per_iteration: int = 8
    max_trail_attempts_per_iteration: int = 10

    # rollout configuration
    student_rollout_batch_size: int = 8

    # Training loop configuration
    epochs_per_step: int = 1
    mini_batch_size: int = 4
    logits_minibatch_size: int = 16

    # Training configuration
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0

    # solved rate conversion threshold
    solved_threshold: float = 1.0

    # misc
    log_interval: int = 1

    num_training_steps: int = 100


@dataclass
class GRPOTrainingConfig(TrainingConfig):
    """Configuration for GRPO training.

    Attributes:
        model_name_or_path: HuggingFace model identifier or local path.
        teacher_group_size: Number of curriculum samples per seed task (G_teacher).
        student_group_size: Number of answer samples per curriculum (G_student).
        max_new_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling probability threshold.
        learning_rate: Optimizer learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        epochs_per_step: Number of optimization epochs per rollout batch.
        clip_eps: PPO clipping epsilon for importance ratio.
        kl_beta: KL penalty coefficient (higher = more conservative updates).
    """

    # GRPO hyperparameters
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    normalize_advantages: bool = True


@dataclass
class TrainingMetrics:
    """Metrics collected during a training update step.

    Attributes:
        loss: Average loss across all batches.
        mean_kl: Average KL divergence from reference policy.
        policy_loss: Average policy loss component.
        grad_norm: Average gradient norm after clipping.
    """

    loss: float
    mean_kl: float
    policy_loss: float
    grad_norm: float

    def to_dict(self, prefix: str = "train/") -> dict[str, float]:
        """Convert metrics to dictionary format.

        Args:
            prefix: Prefix to add to each key (e.g., "train/", "student/train/").

        Returns:
            Dictionary with prefixed keys.
        """
        return {
            f"{prefix}loss": self.loss,
            f"{prefix}mean_kl": self.mean_kl,
            f"{prefix}policy_loss": self.policy_loss,
            f"{prefix}grad_norm": self.grad_norm,
        }
