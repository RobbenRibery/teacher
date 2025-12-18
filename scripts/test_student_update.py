"""Test script for student update."""

from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from teacher_torch.train_grpo import GRPOTrainer, GRPOConfig
from teacher_torch.sampling import SamplingOutput


@dataclass
class StudentUpdateArgs:
    """Arguments for testing student update."""

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    student_group_size: int = 4
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    epochs_per_step: int = 1
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    mini_batch_size: int = 2
    logits_minibatch_size: int = 128
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


def main(args: StudentUpdateArgs = tyro.cli(StudentUpdateArgs)) -> None:
    """Test student update."""
    print("=" * 60)
    print("Testing Student Update")
    print("=" * 60)

    # Create config
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        student_group_size=args.student_group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        epochs_per_step=args.epochs_per_step,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        mini_batch_size=args.mini_batch_size,
        logits_minibatch_size=args.logits_minibatch_size,
    )

    # Create trainer
    trainer = GRPOTrainer(config, seed=args.seed)

    # Get initial model state for comparison
    initial_params = {
        name: param.clone() for name, param in trainer.model.named_parameters()
    }

    # Run student rollout to get experience
    print(f"\nCurriculum:\n{args.curriculum}\n")
    print("Running student rollout...")
    student_exp, student_rollout_metrics = trainer.rollout_student(
        prompts=[args.curriculum],
        student_reward_fn=dummy_student_reward_fn,
    )

    print(f"Rollout metrics:")
    print(f"  Mean reward: {student_rollout_metrics.mean_reward:.4f}")
    print(f"  Mean entropy: {student_rollout_metrics.mean_entropy:.4f}")

    # Run student update
    print("\nRunning student update...")
    from teacher_torch.policy_gradient import update

    student_train_metrics = update(
        model=trainer.model,
        ref_model=trainer.ref_model,
        optimizer=trainer.optimizer,
        experience=student_exp,
        config=config,
        device=trainer.device,
    )

    print("\n" + "=" * 60)
    print("Update Results:")
    print("=" * 60)
    print(f"Training loss: {student_train_metrics.loss:.6f}")
    print(f"Mean KL: {student_train_metrics.mean_kl:.6f}")
    print(f"Policy loss: {student_train_metrics.policy_loss:.6f}")
    print(f"Grad norm: {student_train_metrics.grad_norm:.6f}")

    # Check if parameters changed
    param_changed = False
    for name, initial_param in initial_params.items():
        current_param = dict(trainer.model.named_parameters())[name]
        if not torch.equal(initial_param, current_param):
            param_changed = True
            break

    print(f"\nModel parameters updated: {param_changed}")


if __name__ == "__main__":
    main()
