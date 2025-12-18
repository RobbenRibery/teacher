"""Test script for teacher rollout."""

from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from environment.reward import reward_curriculum_format_only
from teacher_torch.experience import TeacherExperience
from teacher_torch.train_grpo import GRPOTrainer, GRPOConfig
from teacher_torch.sampling import SamplingOutput
from teacher_torch.rollout import RolloutMetric
from teacher_torch.utils import configure_torch_for_training
from curriculum.tasks import TASKS
from pathlib import Path

# Configure torch for bfloat16 training with dynamo optimizations
configure_torch_for_training()


@dataclass
class TeacherRolloutArgs:
    """Arguments for testing teacher rollout."""

    model_name_or_path: str = "Qwen/Qwen3-4B-Thinking-2507"
    teacher_group_size: int = 4
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    min_p: float = 0.0
    seed: int = 42
    repetition_penalty: float = 1


def main(args: TeacherRolloutArgs = tyro.cli(TeacherRolloutArgs)) -> None:
    """Test teacher rollout."""
    print("=" * 60)
    print("Testing Teacher Rollout")
    print("=" * 60)

    # Create config
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        teacher_group_size=args.teacher_group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        # repetition_penalty=args.repetition_penalty,
    )

    # Create teacher generation config with sampling parameters
    teacher_generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        do_sample=True,
        num_return_sequences=args.teacher_group_size,
        cache_implementation="dynamic",
        repetition_penalty=args.repetition_penalty,
        stop_strings=["</answer>", "Human: ", "</answer>\n"],
    )

    # Create trainer
    trainer = GRPOTrainer(
        config, teacher_generation_config=teacher_generation_config, seed=args.seed
    )

    # Test teacher rollout
    print(f"Teacher group size: {args.teacher_group_size}")
    print("Running teacher rollout...")

    teacher_exp, teacher_metrics = trainer.rollout_teacher(
        teacher_reward_fn=reward_curriculum_format_only,
    )
    print(f"Prompt: {teacher_exp.prompt}")
    print(f"Number of completions: {len(teacher_exp.completions)}")

    # Save teacher experience
    _dir_ = Path(f"data/teacher_0")
    _dir_.mkdir(parents=True, exist_ok=True)
    with open(_dir_ / "prompt.txt", "w") as f:
        f.write(teacher_exp.prompt)
    for j, completion in enumerate(teacher_exp.completions):
        with open(_dir_ / f"completion_{j}.txt", "w") as f:
            f.write(completion)
        if teacher_exp.rewards is not None:
            with open(_dir_ / f"reward_{j}.txt", "w") as f:
                f.write(str(teacher_exp.rewards[j].item()))


if __name__ == "__main__":
    main()
