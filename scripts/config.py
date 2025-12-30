"""Centralized configuration parsing using tyro."""

from __future__ import annotations

import tyro
from dataclasses import dataclass, field

from teacher_torch.configs import (
    GRPOTrainingConfig,
    TeacherGenerationConfig,
    StudentGenerationConfig,
)


def _default_teacher_gen_config(seed: int = 42) -> TeacherGenerationConfig:
    """Create default teacher generation config."""
    return TeacherGenerationConfig(
        teacher_group_size=8,
        max_new_tokens=8192,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=seed,
    )


def _default_student_gen_config(seed: int = 42) -> StudentGenerationConfig:
    """Create default student generation config."""
    return StudentGenerationConfig(
        student_group_size=16,
        max_new_tokens=4096,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=seed,
    )


def _default_config() -> GRPOTrainingConfig:
    """Create default GRPOTrainingConfig."""
    return GRPOTrainingConfig(
        mode_name_or_path="Qwen/Qwen3-1.7B",
        teacher_generation_config=_default_teacher_gen_config(),
        student_generation_config=_default_student_gen_config(),
    )


@dataclass
class ConfigArgs:
    """Command-line arguments wrapper for GRPO training configuration.
    
    Uses tyro's nested config support to parse GRPOTrainingConfig directly.
    Adds seed at the top level since trainer accesses train_config.seed.
    """
    
    # Seed for reproducibility (accessed by trainer)
    seed: int = 42
    
    # Nested GRPO training config (tyro handles nested parsing automatically)
    # Using field(default_factory) to create default nested configs
    config: GRPOTrainingConfig = field(default_factory=_default_config)
    
    def __post_init__(self):
        """Ensure seed is synchronized across config."""
        # Update seed in generation configs
        self.config.teacher_generation_config.seed = self.seed
        self.config.student_generation_config.seed = self.seed
        # Add seed to config for trainer access
        self.config.seed = self.seed


def parse_args() -> tuple[GRPOTrainingConfig, int]:
    """Parse command-line arguments and return GRPOTrainingConfig and seed.
    
    Uses tyro's nested config support to parse GRPOTrainingConfig directly.
    
    Returns:
        Tuple of (GRPOTrainingConfig instance, seed value).
    """
    args = tyro.cli(ConfigArgs)
    return args.config, args.seed

