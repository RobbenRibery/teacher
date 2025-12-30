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
        max_new_tokens=4096,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.25,
        seed=seed,
        do_sample=True,
        teacher_group_size=8,
    )


def _default_student_gen_config(seed: int = 42) -> StudentGenerationConfig:
    """Create default student generation config."""
    return StudentGenerationConfig(
        max_new_tokens=4096,
        temperature=0.8,
        top_k=20,
        top_p=0.95,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=seed,
        student_group_size=16,
    )


def _default_config() -> GRPOTrainingConfig:
    """Create default GRPOTrainingConfig."""
    return GRPOTrainingConfig(
        mode_name_or_path="Qwen/Qwen3-1.7B",
        teacher_generation_config=_default_teacher_gen_config(),
        student_generation_config=_default_student_gen_config(),
        minimal_legit_curriculum_per_iteration=32,
        max_trail_attempts_per_iteration=10,
    )


@dataclass
class ConfigArgs:
    """Command-line arguments wrapper for GRPO training configuration.
    
    Uses tyro's nested config support to parse GRPOTrainingConfig directly.
    """
    config: GRPOTrainingConfig = field(default_factory=_default_config)


def parse_args() -> GRPOTrainingConfig:
    """Parse command-line arguments and return GRPOTrainingConfig and seed.
    
    Uses tyro's nested config support to parse GRPOTrainingConfig directly.
    
    Returns:
        Tuple of (GRPOTrainingConfig instance, seed value).
    """
    args = tyro.cli(ConfigArgs)
    return args.config

