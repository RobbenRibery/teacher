"""Test script for student rollout."""
from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from teacher_torch.train_grpo import GRPOTrainer, GRPOConfig
from teacher_torch.sampling import SamplingOutput


@dataclass
class StudentRolloutArgs:
    """Arguments for testing student rollout."""
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    student_group_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    curriculum: str = """
        Find an integer x such that:
        - x > 5
        - x < 10
        - x % 2 == 0
    """


def dummy_student_reward_fn(sampling_output: SamplingOutput) -> dict:
    """Dummy reward function for testing."""
    # Return random rewards for testing
    num_samples = len(sampling_output.completions)
    rewards = torch.rand(num_samples) * 2.0  # Random rewards between 0 and 2
    return {"rewards": rewards}


def main(args: StudentRolloutArgs = tyro.cli(StudentRolloutArgs)) -> None:
    """Test student rollout."""
    print("=" * 60)
    print("Testing Student Rollout")
    print("=" * 60)
    
    # Create config
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        student_group_size=args.student_group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Create trainer
    trainer = GRPOTrainer(config, seed=args.seed)
    
    # Test student rollout
    print(f"\nCurriculum:\n{args.curriculum}\n")
    print(f"Student group size: {args.student_group_size}")
    print("Running student rollout...")
    
    student_exp, student_metrics = trainer.rollout_student(
        prompts=[args.curriculum],
        student_reward_fn=dummy_student_reward_fn,
    )
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Number of experiences: {len(student_exp.experiences)}")
    print(f"Mean reward: {student_metrics.mean_reward:.4f}")
    print(f"Std reward: {student_metrics.std_reward:.4f}")
    print(f"Mean entropy: {student_metrics.mean_entropy:.4f}")
    print(f"Mean completion length: {student_metrics.mean_completion_len:.2f}")
    
    if student_exp.experiences:
        exp = student_exp.experiences[0]
        print(f"\nFirst experience:")
        print(f"  Returns shape: {exp.returns.shape}")
        print(f"  Advantages shape: {exp.advantages.shape}")
        print(f"  Number of completions: {len(exp.completion)}")
        if exp.completion:
            print(f"\nFirst completion (first 200 chars):")
            print(f"  {exp.completion[0][:200]}...")


if __name__ == "__main__":
    main()

