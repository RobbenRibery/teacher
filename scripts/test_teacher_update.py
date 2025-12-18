"""Test script for teacher update."""

from __future__ import annotations

import torch
from transformers import GenerationConfig
import tyro
from dataclasses import dataclass

from teacher_torch.train_grpo import GRPOTrainer, GRPOConfig
from teacher_torch.sampling import SamplingOutput


@dataclass
class TeacherUpdateArgs:
    """Arguments for testing teacher update."""

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    teacher_group_size: int = 4
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


def main(args: TeacherUpdateArgs = tyro.cli(TeacherUpdateArgs)) -> None:
    """Test teacher update."""
    print("=" * 60)
    print("Testing Teacher Update")
    print("=" * 60)

    # Create config
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        teacher_group_size=args.teacher_group_size,
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

    # Run teacher rollout to get experience
    print(f"\nSeed task:\n{args.seed_task}\n")
    print("Running teacher rollout...")
    teacher_exp, teacher_rollout_metrics = trainer.rollout_teacher(
        seed_tasks=[args.seed_task],
        teacher_reward_fn=dummy_teacher_reward_fn,
    )

    print(f"Rollout metrics:")
    print(f"  Mean reward: {teacher_rollout_metrics.mean_reward:.4f}")
    print(f"  Mean entropy: {teacher_rollout_metrics.mean_entropy:.4f}")

    # Run teacher update
    print("\nRunning teacher update...")
    from teacher_torch.policy_gradient import update

    teacher_train_metrics = update(
        model=trainer.model,
        ref_model=trainer.ref_model,
        optimizer=trainer.optimizer,
        experience=teacher_exp,
        config=config,
        device=trainer.device,
    )

    print("\n" + "=" * 60)
    print("Update Results:")
    print("=" * 60)
    print(f"Training loss: {teacher_train_metrics.loss:.6f}")
    print(f"Mean KL: {teacher_train_metrics.mean_kl:.6f}")
    print(f"Policy loss: {teacher_train_metrics.policy_loss:.6f}")
    print(f"Grad norm: {teacher_train_metrics.grad_norm:.6f}")

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
