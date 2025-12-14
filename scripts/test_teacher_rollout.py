"""Test script for teacher rollout."""
from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from environment.reward import (
    reward_curriculum_format_only
)
from teacher_torch.experience import ExperienceBuffer
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
    model_name_or_path: str = "Qwen/Qwen3-4B"
    teacher_group_size: int = 4
    max_new_tokens: int = 8192
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    seed: int = 42

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
    )
    
    # Create teacher generation config with sampling parameters
    teacher_generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        num_return_sequences=args.teacher_group_size,
        cache_implementation="dynamic",
        repetition_penalty=1.3,
    )
    
    # Create trainer
    trainer = GRPOTrainer(config, teacher_generation_config=teacher_generation_config, seed=args.seed)
    
    # Test teacher rollout
    print(f"Teacher group size: {args.teacher_group_size}")
    print("Running teacher rollout...")
    
    teacher_exp, teacher_metrics = trainer.rollout_teacher(
        teacher_reward_fn=reward_curriculum_format_only,
    )
    print(f"Number of unique prompts: {len(teacher_exp.prompts)}")
    print(f"Number of completions: {len(teacher_exp.completions)}")

    for i, exp in enumerate(teacher_exp):
        _dir_ = Path(f"data/teacher_{i}")
        _dir_.mkdir(parents=True, exist_ok=True)
        with open(_dir_ / "prompt.txt", "w") as f:
            f.write(exp.prompt)
        for j, completion in enumerate(exp.completions):
            with open(_dir_ / f"completion_{j}.txt", "w") as f:
                f.write(completion)
            with open(_dir_ / f"reward_{j}.txt", "w") as f:
                f.write(str(exp.rewards[j]))

if __name__ == "__main__":
    main()
