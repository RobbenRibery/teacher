"""Teacher rollout script with centralized config and wandb logging."""

from __future__ import annotations

import wandb
from pathlib import Path
from dataclasses import asdict

from scripts.config import parse_args
from teacher_torch.trainer import GRPOTrainer
from teacher_torch.utils import configure_torch_for_training
from environment.reward import reward_curriculum_format_only


def main() -> None:
    """Run teacher rollout with config from CLI and log to wandb."""
    # Parse configuration from command line
    config, seed = parse_args()
    
    # Configure torch for bfloat16 training with dynamo optimizations
    configure_torch_for_training()
    
    # Prepare config dict for wandb (convert nested dataclasses to dicts)
    config_dict = asdict(config)
    config_dict["teacher_generation_config"] = asdict(config.teacher_generation_config)
    config_dict["student_generation_config"] = asdict(config.student_generation_config)
    config_dict["seed"] = seed  # Add seed explicitly
    
    # Initialize wandb with config
    wandb.init(
        project="teacher-rollout",
        config=config_dict,
        name=f"teacher_rollout_seed_{seed}",
    )
    
    print("=" * 60)
    print("Teacher Rollout")
    print("=" * 60)
    print(f"Model: {config.mode_name_or_path}")
    print(f"Teacher group size: {config.teacher_group_size}")
    print(f"Seed: {seed}")
    
    # Create trainer
    trainer = GRPOTrainer(config)
    
    # Run teacher rollout
    print("Running teacher rollout...")
    teacher_exp, teacher_metrics = trainer.rollout_teacher(
        teacher_reward_fn=reward_curriculum_format_only,
    )
    
    print(f"Prompt: {teacher_exp.prompt}")
    print(f"Number of completions: {len(teacher_exp.completions)}")
    
    # Log metrics to wandb
    metrics_dict = {
        "rollout/mean_reward": teacher_metrics.mean_reward,
        "rollout/std_reward": teacher_metrics.std_reward,
        "rollout/mean_entropy": teacher_metrics.mean_entropy,
        "rollout/num_completions": len(teacher_exp.completions),
    }
    if teacher_exp.rewards is not None:
        metrics_dict["rollout/mean_reward_from_exp"] = teacher_exp.rewards.mean().item()
        metrics_dict["rollout/std_reward_from_exp"] = teacher_exp.rewards.std().item()
    
    wandb.log(metrics_dict)
    
    # Save teacher experience to disk
    output_dir = Path("data/teacher")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "prompt.txt", "w") as f:
        f.write(teacher_exp.prompt)
    
    for j, completion in enumerate(teacher_exp.completions):
        with open(output_dir / f"completion_{j}.txt", "w") as f:
            f.write(completion)
        
        if teacher_exp.rewards is not None:
            with open(output_dir / f"reward_{j}.txt", "w") as f:
                f.write(str(teacher_exp.rewards[j].item()))
    
    print(f"Results saved to {output_dir}")
    print("Rollout complete!")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

