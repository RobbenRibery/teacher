"""Teacher rollout script with centralized config and wandb logging."""

from __future__ import annotations

import wandb
import torch
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from scripts.config import parse_args
from teacher_torch.trainer import GRPOTrainer
from teacher_torch.utils import configure_torch_for_training
from environment.reward import function_reward_curriculum_format_only
import rich
import json


def main() -> None:
    """Run teacher rollout with config from CLI and log to wandb."""
    # Parse configuration from command line
    config = parse_args()

    # Configure torch for bfloat16 training with dynamo optimizations
    configure_torch_for_training()

    # Prepare config dict for wandb (convert nested dataclasses to dicts)
    config_dict = asdict(config)

    # Initialize wandb with config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="teacher-rollout",
        config=config_dict,
        name=timestamp,
        tags=["teacher_rollout"],
    )

    rich.print("=" * 60)
    rich.print("Teacher Rollout")
    rich.print("=" * 60)
    rich.print(f"Model: {config.mode_name_or_path}")
    rich.print(
        f"Teacher group size: {config.teacher_generation_config.teacher_group_size}"
    )

    # Create trainer
    trainer = GRPOTrainer(config)
    output_dir = Path("data/teacher") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    rich.print(f"Config saved to {output_dir / 'config.json'}")

    # Run teacher rollout
    rich.print("Running teacher rollout...")
    teacher_exp, teacher_metrics = trainer.rollout_teacher(
        teacher_reward_fn=function_reward_curriculum_format_only,
        output_dir=output_dir,
    )

    rich.print(f"Prompt: {teacher_exp.prompt}")
    rich.print(f"Number of completions: {len(teacher_exp.completions)}")

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

    with open(output_dir / "prompt.txt", "w") as f:
        f.write(teacher_exp.prompt)

    for j, completion in enumerate(teacher_exp.completions):
        with open(output_dir / f"completion_{j}.txt", "w") as f:
            f.write(completion)

        if teacher_exp.rewards is not None:
            with open(output_dir / f"reward_{j}.txt", "w") as f:
                f.write(str(teacher_exp.rewards[j].item()))

    rich.print(f"Results saved to {output_dir}")

    # Save TeacherExperience as JSON for student_rollout.py
    teacher_exp_dict = teacher_exp.to_dict()
    teacher_exp_json_path = Path("scripts/teacher_exp.json")
    with open(teacher_exp_json_path, "w") as f:
        json.dump(teacher_exp_dict, f, indent=4)
    rich.print(f"TeacherExperience saved to {teacher_exp_json_path}")

    # Also save a copy in output_dir for archival
    with open(output_dir / "teacher_exp.json", "w") as f:
        json.dump(teacher_exp_dict, f, indent=4)

    rich.print("Rollout complete!")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
