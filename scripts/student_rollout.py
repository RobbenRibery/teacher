"""Student rollout script with centralized config and wandb logging.

Loads a TeacherExperience from JSON and runs student rollout on the curriculums.
"""

from __future__ import annotations

import wandb
import json
import torch
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
from scripts.config import parse_args
from teacher_torch.trainer import GRPOTrainer
from teacher_torch.utils import configure_torch_for_training
from teacher_torch.experience import TeacherExperience
from environment.reward import function_reward_answer
from environment.format import STUDENT_PATTERN
import rich


def main() -> None:
    """Run student rollout with TeacherExperience from JSON and log to wandb."""
    # Parse configuration from command line
    config = parse_args()
    
    # Configure torch for bfloat16 training with dynamo optimizations
    configure_torch_for_training()
    
    # Load TeacherExperience from JSON
    teacher_exp_path = Path("scripts/teacher_exp.json")
    if not teacher_exp_path.exists():
        raise FileNotFoundError(
            f"TeacherExperience JSON not found at {teacher_exp_path}. "
            "Please run teacher_rollout.py first and save the experience."
        )
    
    with open(teacher_exp_path, "r") as f:
        teacher_data = json.load(f)
    teacher_exp = TeacherExperience.from_dict(teacher_data)
    
    # Extract curriculums from the loaded experience
    curriculums = teacher_exp.curriculums
    if not curriculums:
        raise ValueError(
            "No curriculums found in TeacherExperience JSON. "
            "Make sure the JSON contains a 'curriculums' field with valid curriculum strings."
        )
    
    # Filter out None curriculums
    valid_curriculums = [c for c in curriculums if c is not None]
    if not valid_curriculums:
        raise ValueError("No valid (non-None) curriculums found in TeacherExperience JSON.")
    
    rich.print(f"Loaded {len(valid_curriculums)} valid curriculums from {teacher_exp_path}")
    
    # Prepare config dict for wandb (convert nested dataclasses to dicts)
    config_dict = asdict(config)
    config_dict["num_curriculums"] = len(valid_curriculums)
    
    # Initialize wandb with config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.init(
        project="student-rollout",
        config=config_dict,
        name=timestamp,
        tags=["student_rollout"],
    )
    
    rich.print("=" * 60)
    rich.print("Student Rollout")
    rich.print("=" * 60)
    rich.print(f"Model: {config.mode_name_or_path}")
    rich.print(f"Student group size: {config.student_generation_config.student_group_size}")
    rich.print(f"Number of curriculums: {len(valid_curriculums)}")
    
    # Create trainer
    trainer = GRPOTrainer(config)
    output_dir = Path("data/student") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    rich.print(f"Config saved to {output_dir / 'config.json'}")

    
    # Run student rollout
    rich.print("Running student rollout...")
    student_buffer, student_metrics = trainer.rollout_student(
        curriculums=valid_curriculums,
        student_reward_fn=function_reward_answer,
        output_dir=output_dir,
    )
    
    rich.print(f"Number of student experiences: {len(student_buffer)}")
    rich.print(f"Number of unique prompts: {student_buffer.num_unique_prompts}")
    
    # Log metrics to wandb
    metrics_dict = {
        "rollout/mean_reward": student_metrics.mean_reward,
        "rollout/std_reward": student_metrics.std_reward,
        "rollout/mean_entropy": student_metrics.mean_entropy,
        "rollout/mean_solved_rate": student_metrics.mean_solved_rate or 0.0,
        "rollout/mean_learnability": student_metrics.mean_learnability or 0.0,
        "rollout/num_experiences": len(student_buffer),
        "rollout/num_completions": len(student_buffer.completions),
    }
    
    wandb.log(metrics_dict)
    
    # Save prompts and completions
    for i, exp in enumerate(student_buffer):
        exp_dir = output_dir / f"curriculum_{i}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save source curriculum
        if exp.source_curriculum:
            with open(exp_dir / "source_curriculum.txt", "w") as f:
                f.write(exp.source_curriculum)
        
        # Save prompt
        with open(exp_dir / "prompt.txt", "w") as f:
            f.write(exp.prompt)
        
        # Save each completion and its reward
        for j, completion in enumerate(exp.completions):
            with open(exp_dir / f"completion_{j}.txt", "w") as f:
                f.write(completion)
            
            if exp.rewards is not None:
                with open(exp_dir / f"reward_{j}.txt", "w") as f:
                    f.write(str(exp.rewards[j].item()))
        
        # Save solved rate for this curriculum
        with open(exp_dir / "solved_rate.txt", "w") as f:
            f.write(str(exp.solved_rate))
    
    # Save summary
    summary = {
        "num_curriculums": len(valid_curriculums),
        "num_experiences": len(student_buffer),
        "mean_reward": student_metrics.mean_reward,
        "std_reward": student_metrics.std_reward,
        "mean_solved_rate": student_metrics.mean_solved_rate,
        "mean_learnability": student_metrics.mean_learnability,
        "solved_rates": [exp.solved_rate for exp in student_buffer],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    rich.print(f"\nResults saved to {output_dir}")
    rich.print(f"Mean reward: {student_metrics.mean_reward:.4f}")
    rich.print(f"Mean solved rate: {student_metrics.mean_solved_rate:.4f}")
    rich.print("Student rollout complete!")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()

