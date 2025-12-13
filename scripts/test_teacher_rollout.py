"""Test script for teacher rollout."""
from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from teacher_torch.experience import ExperienceBuffer
from teacher_torch.train_grpo import GRPOTrainer, GRPOConfig
from teacher_torch.sampling import SamplingOutput
from teacher_torch.rollout import RolloutMetric
from curriculum.tasks import TASKS

@dataclass
class TeacherRolloutArgs:
    """Arguments for testing teacher rollout."""
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    teacher_group_size: int = 4
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
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
    )
    
    # Create trainer
    trainer = GRPOTrainer(config, teacher_generation_config=teacher_generation_config, seed=args.seed)
    
    # Test teacher rollout
    print(f"Teacher group size: {args.teacher_group_size}")
    print("Running teacher rollout...")
    
    teacher_exp, teacher_metrics = trainer.rollout_teacher()

if __name__ == "__main__":
    main()

