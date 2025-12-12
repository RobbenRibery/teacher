"""Test script for teacher rollout."""
from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from teacher_torch.train_grpo import GRPOTrainer, GRPOConfig
from teacher_torch.sampling import SamplingOutput


@dataclass
class TeacherRolloutArgs:
    """Arguments for testing teacher rollout."""
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    teacher_group_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    seed_task: str = """
        Design a problem where the student must find an integer x that satisfies:
        - Mathematical constraints involving arithmetic operations
        - The solution should require multi-step reasoning
    """


def dummy_teacher_reward_fn(sampling_output: SamplingOutput) -> dict:
    """Dummy reward function for testing."""
    # Return random rewards for testing
    num_samples = len(sampling_output.completions)
    rewards = torch.rand(num_samples) * 2.0  # Random rewards between 0 and 2
    return {"rewards": rewards}


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
        top_p=args.top_p,
    )
    
    # Create trainer
    trainer = GRPOTrainer(config, seed=args.seed)
    
    # Test teacher rollout
    print(f"\nSeed task:\n{args.seed_task}\n")
    print(f"Teacher group size: {args.teacher_group_size}")
    print("Running teacher rollout...")
    
    teacher_exp, teacher_metrics = trainer.rollout_teacher(
        seed_tasks=[args.seed_task],
        teacher_reward_fn=dummy_teacher_reward_fn,
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Number of experiences: {len(teacher_exp.experiences)}")
    print(f"Mean reward: {teacher_metrics.mean_reward:.4f}")
    print(f"Std reward: {teacher_metrics.std_reward:.4f}")
    print(f"Mean entropy: {teacher_metrics.mean_entropy:.4f}")
    print(f"Mean completion length: {teacher_metrics.mean_completion_len:.2f}")
    
    if teacher_exp.experiences:
        exp = teacher_exp.experiences[0]
        print(f"\nFirst experience:")
        print(f"  Returns shape: {exp.returns.shape}")
        print(f"  Advantages shape: {exp.advantages.shape}")
        print(f"  Number of completions: {len(exp.completion)}")
        if exp.completion:
            print(f"\nFirst completion (first 200 chars):")
            print(f"  {exp.completion[0][:200]}...")


if __name__ == "__main__":
    main()

