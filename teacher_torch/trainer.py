"""
GRPO Training Loop for Teacher-Student Curriculum Learning.

This module implements a training pipeline where:
1. A Teacher model generates curricula (problems/tasks)
2. A Student model attempts to solve these curricula
3. Both receive rewards based on performance
4. Both are trained using GRPO with shared model weights

Key features:
- Shared model between Teacher and Student roles
- KL regularization against a frozen reference model
- Group-relative advantage normalization (no value function needed)
- Async reward computation for efficiency
"""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass
from typing import Optional, Any, Callable

import torch
from transformers import GenerationConfig

from teacher_torch.experience import TeacherExperience, StudentExperienceBuffer
from teacher_torch.rollout import rollout, RolloutMetric, RolloutConfig
from teacher_torch.sampling import SamplingOutput
from teacher_torch.utils import (
    setup_logger_with_file_line,
    load_model_tokenizer, 
    split_seed,
)
from teacher_torch.policy_gradient import update 
from teacher_torch.configs import (
    GRPOTrainingConfig,
    TrainingMetrics,
)

from environment.reward import CurriculumReward, AnswerReward

from curriculum.prompt_template import TeacherPrompt, StudentPrompt
from curriculum.tasks import BooleanConstraintsSatisfaction

import wandb
from pathlib import Path
logger = setup_logger_with_file_line(__name__)

@dataclass
class RunnerMetrics:
    """Metrics collected during a training step.

    Combines metrics from rollout phases and training updates.

    Attributes:
        step: Training step number.
        teacher_mean_reward: Mean reward from teacher rollout.
        teacher_std_reward: Standard deviation of teacher rewards.
        teacher_mean_entropy: Mean entropy from teacher rollout.
        student_mean_reward: Mean reward from student rollout.
        student_std_reward: Standard deviation of student rewards.
        student_mean_entropy: Mean entropy from student rollout.
        student_train_loss: Training loss from student update.
        student_train_mean_kl: Mean KL divergence from student update.
        student_train_policy_loss: Policy loss from student update.
        student_train_grad_norm: Gradient norm from student update.
        teacher_train_loss: Training loss from teacher update.
        teacher_train_mean_kl: Mean KL divergence from teacher update.
        teacher_train_policy_loss: Policy loss from teacher update.
        teacher_train_grad_norm: Gradient norm from teacher update.
    """

    step: int
    teacher_mean_reward: float
    teacher_std_reward: float
    teacher_mean_entropy: float
    student_mean_reward: float
    student_std_reward: float
    student_mean_entropy: float
    student_train_loss: float
    student_train_mean_kl: float
    student_train_policy_loss: float
    student_train_grad_norm: float
    teacher_train_loss: float
    teacher_train_mean_kl: float
    teacher_train_policy_loss: float
    teacher_train_grad_norm: float

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary format for logging.

        Returns:
            Dictionary with all metrics as key-value pairs.
        """
        return {
            "step": self.step,
            "teacher/mean_reward": self.teacher_mean_reward,
            "teacher/std_reward": self.teacher_std_reward,
            "teacher/mean_entropy": self.teacher_mean_entropy,
            "student/mean_reward": self.student_mean_reward,
            "student/std_reward": self.student_std_reward,
            "student/mean_entropy": self.student_mean_entropy,
            "student/train/loss": self.student_train_loss,
            "student/train/mean_kl": self.student_train_mean_kl,
            "student/train/policy_loss": self.student_train_policy_loss,
            "student/train/grad_norm": self.student_train_grad_norm,
            "teacher/train/loss": self.teacher_train_loss,
            "teacher/train/mean_kl": self.teacher_train_mean_kl,
            "teacher/train/policy_loss": self.teacher_train_policy_loss,
            "teacher/train/grad_norm": self.teacher_train_grad_norm,
        }


class GRPOTrainer:
    """GRPO Trainer for Teacher-Student Curriculum Learning.

    Implements a training loop where a shared model plays both roles:
    - As Teacher: generates curricula (problems) from seed task descriptions
    - As Student: attempts to solve the generated curricula

    The training objective encourages:
    - Teacher to generate curricula at appropriate difficulty (not too easy/hard)
    - Student to solve curricula correctly
    - Both to stay close to reference policy (KL regularization)

    Attributes:
        config: Training configuration.
        device: Compute device (CPU/GPU).
        model: The shared language model being trained.
        tokenizer: Tokenizer for the model.
        ref_model: Frozen reference model for KL computation.
        optimizer: AdamW optimizer.
        task: Task definition (e.g., BooleanConstraintsSatisfaction).
        teacher_prompt_template: Template for teacher prompts.
        format_regex: Regex for extracting code from generations.

    Example:
        >>> config = GRPOConfig(model_name_or_path="Qwen/Qwen3-0.6B")
        >>> trainer = GRPOTrainer(config)
        >>> metrics = trainer.train(seed_tasks, num_steps=100)
    """

    def __init__(self, train_config: GRPOTrainingConfig) -> None:
        """Initialize the GRPO trainer.

        Args:
            train_config: Training configuration object.
        """
        self.train_config = train_config

        # Load or use provided model/tokenizer
        self.model, self.tokenizer = load_model_tokenizer(
            train_config.mode_name_or_path,
            device_map="auto" if torch.cuda.is_available() else None,
            #chat_template_path="template_lm_studio.jinja",
        )

        # Create frozen reference model for KL regularization
        self.ref_model, _ = load_model_tokenizer(
            train_config.mode_name_or_path,
            device_map="auto" if torch.cuda.is_available() else None,
            #chat_template_path="template_lm_studio.jinja",
        )
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.learning_rate,
        )

        # Task and prompt templates
        self.task = BooleanConstraintsSatisfaction()
        self.teacher_prompt_template = TeacherPrompt()
        self.student_prompt_template = StudentPrompt()

        # generation config
        self.teacher_generation_config = train_config.teacher_generation_config
        self.student_generation_config = train_config.student_generation_config

    def _build_teacher_prompt(self) -> str:
        """Build formatted teacher prompt from seed task.

        Args:
            seed_task: High-level task description for curriculum generation.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        prompt = self.teacher_prompt_template.prompt(
            seed_task=self.task.render(),
            num_instances=1,
        )
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _build_student_prompt(self, problem: str) -> str:
        """Build formatted student prompt from curriculum.

        Args:
            problem: The problem for the student to solve.

        Returns:
            Formatted prompt string ready for tokenization.
        """
        prompt = self.student_prompt_template.prompt(problem=problem)
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def rollout_teacher(
        self,
        teacher_reward_fn: Callable[[str], CurriculumReward],
        output_dir: Optional[Path] = None,  
    ) -> tuple[TeacherExperience, RolloutMetric]:
        """Execute teacher rollout phase.

        Generates multiple curricula for a single seed task using the provided
        reward function. If min_legit_problems > 0, keeps rolling out until
        we accumulate that many correctly formatted curricula (or hit max attempts).

        Args:
            teacher_reward_fn: Callable that takes completion string and returns Reward object.
            output_dir: Directory to save the rollout output.
        Returns:
            Tuple containing:
                - experience: TeacherExperience with generated curricula.
                - metrics: RolloutMetric containing generation statistics (from last attempt).
        """
        if not teacher_reward_fn:
            raise ValueError("Teacher reward function is required")

        prompts = [self._build_teacher_prompt()]

        def do_rollout(seed: int) -> tuple[TeacherExperience, RolloutMetric]:
            rollout_config = RolloutConfig(
                seed=seed,
                normalize_advantages=self.train_config.normalize_advantages,
            )
            return rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                generation_config=GenerationConfig(**self.teacher_generation_config.to_dict()),
                rollout_config=rollout_config,
                role="teacher",
                solved_threshold=self.train_config.solved_threshold,
                reward_fn=teacher_reward_fn,
                logits_minibatch_size=self.train_config.logits_minibatch_size,
                only_check_format=True,
                ref_model=self.ref_model,
                output_dir=output_dir,
            )

        min_legit = self.train_config.minimal_legit_curriculum_per_iteration
        max_attempts = self.train_config.max_trail_attempts_per_iteration

        # Single rollout if no filtering needed
        if min_legit <= 0:
            return do_rollout(seed=self.teacher_generation_config.seed)

        # Accumulate correctly formatted curricula
        curated: Optional[TeacherExperience] = None
        experience, metrics = None, None
        current_seed = self.teacher_generation_config.seed

        for idx in range(max_attempts):
            num_curricula = curated.num_curricula if curated is not None else 0
            logger.info(f"========= Rolling out teacher... {idx + 1}/{max_attempts} | Accumulating {num_curricula}/{min_legit} curricula ==========")
            experience, metrics = do_rollout(seed=current_seed)

            correct_formatted_experience = experience.correct_formatted_experience
            logger.info(f"Number of correctly formatted curricula: {correct_formatted_experience.num_curricula if correct_formatted_experience is not None else 0}")
            
            # Update seed for next iteration
            current_seed = split_seed(current_seed)

            if correct_formatted_experience is not None:
                if curated is None:
                    curated = correct_formatted_experience
                else:
                    curated = curated.merge(correct_formatted_experience)

                if curated is not None:
                    logger.info(f"Number of curricula after merging: {curated.num_curricula}")
                    if curated.num_curricula >= min_legit:
                        break

        # Save final seed back to config for subsequent calls
        self.teacher_generation_config.seed = current_seed
        output_exp = curated or experience

        return output_exp, metrics

    def rollout_student(
        self,
        curriculums: list[str],
        student_reward_fn: Callable[[tuple[str,str]], AnswerReward],
        output_dir: Optional[Path] = None,
    ) -> tuple[StudentExperienceBuffer, RolloutMetric]:
        """Execute student rollout phase.

        For each problem, generates multiple student answer attempts
        and computes rewards using the environment.

        Args:
            curriculums: List of curriculums for the student to solve.
            student_reward_fn: Optional reward function for student answers.

        Returns:
            Tuple containing:
                - experience: StudentExperienceBuffer with generated answers.
                - metrics: RolloutMetric containing generation statistics.
        """
        if not student_reward_fn:
            raise ValueError("Student reward function is required")
        
        # Create rollout config
        rollout_config = RolloutConfig(
            seed=self.student_generation_config.seed,
            normalize_advantages=self.train_config.normalize_advantages,
        )

        # Batch processing to avoid OOM
        # Use mini_batch_size from config or default to 4
        rollout_batch_size = self.train_config.student_rollout_batch_size
        total_curriculums = len(curriculums)
        
        all_buffer = StudentExperienceBuffer()
        total_entropy_sum = 0.0
        total_samples = 0
        
        logger.info(f"Starting student rollout for {total_curriculums} curriculums in batches of {rollout_batch_size}")

        for i in range(0, total_curriculums, rollout_batch_size):
            batch_curriculums = curriculums[i : i + rollout_batch_size]
            logger.info(f"Processing student rollout batch {i//rollout_batch_size + 1}/{(total_curriculums + rollout_batch_size - 1)//rollout_batch_size} ({len(batch_curriculums)} items)")
            
            # Format prompts for student (apply chat template)
            formatted_prompts = [self._build_student_prompt(c) for c in batch_curriculums]

            # Generate answers with source_curriculums for tracking
            # Pass output_dir=None to avoid overwriting files or creating flat list
            # The caller (student_rollout.py) handles saving based on returned buffer
            batch_result, batch_metrics = rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=formatted_prompts,
                generation_config=GenerationConfig(**self.student_generation_config.to_dict()),
                rollout_config=rollout_config,
                role="student",
                solved_threshold=self.train_config.solved_threshold,
                reward_fn=[
                    lambda completion, c=c: student_reward_fn(extracted_curriculum=c, answer_generation=completion)
                    for c in batch_curriculums
                ],
                logits_minibatch_size=self.train_config.logits_minibatch_size,
                only_check_format=False,
                source_curriculums=batch_curriculums,
                ref_model=self.ref_model,
                output_dir=output_dir / f"rollout_batch_{i}"
            )
            
            # Accumulate results
            all_buffer.extend(batch_result)
            total_entropy_sum += batch_metrics.mean_entropy * batch_metrics.num_samples
            total_samples += batch_metrics.num_samples
            
            # Clear CUDA cache after each batch to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Update seed for next time
        self.student_generation_config.seed = split_seed(self.student_generation_config.seed)
        
        # Recompute aggregate metrics from the full buffer
        all_rewards = []
        all_completion_lens = []
        all_solved = []
        all_learnability = []
        all_advantages = []
        
        for exp in all_buffer.experiences:
            if exp.rewards is not None:
                all_rewards.append(exp.rewards)
            if exp.action_mask is not None:
                all_completion_lens.append(exp.action_mask.sum(dim=1).float())
            if exp.solved_mask is not None:
                all_solved.append(exp.solved_mask.float())
            if exp.learnability is not None:
                all_learnability.append(exp.learnability)
            if exp.advantages is not None:
                all_advantages.append(exp.advantages)
        
        # Helper to compute mean/std safely
        def safe_mean(tensors):
            if not tensors: return 0.0
            cat = torch.cat(tensors) if isinstance(tensors[0], torch.Tensor) else torch.tensor(tensors)
            return cat.mean().item()
            
        def safe_std(tensors):
            if not tensors: return 0.0
            cat = torch.cat(tensors) if isinstance(tensors[0], torch.Tensor) else torch.tensor(tensors)
            if cat.numel() > 1:
                return cat.std().item()
            return 0.0
            
        # Reconstruct aggregated metrics
        final_metrics = RolloutMetric(
            num_samples=total_samples,
            group_size=self.student_generation_config.student_group_size, # Assuming constant group size
            mean_entropy=total_entropy_sum / total_samples if total_samples > 0 else 0.0,
            mean_completion_len=safe_mean(all_completion_lens),
            mean_reward=safe_mean(all_rewards),
            std_reward=safe_std(all_rewards),
            mean_solved_rate=safe_mean(all_solved),
            mean_learnability=safe_mean(all_learnability) if isinstance(all_learnability, list) else 0.0, # learnability is scalar per group? Exp has it as scalar or tensor?
            # StudentExperience has learnability as Optional[float].
            # So all_learnability is list[float].
            # Actually StudentExperience definition: learnability: Optional[float] = None
            # So safe_mean works on list[float] too if I adapt it.
            mean_advantages=safe_mean(all_advantages),
        )

        return all_buffer, final_metrics

    def train_step(
        self, teacher_exp: TeacherExperience, student_exps: StudentExperienceBuffer
    ) -> dict[str, float]:
        """Perform a single training step with separate updates for student and teacher.

        Updates are performed in order:
        1. Update teacher rewards based on student performance
        2. Update model using student experience (with KL regularization)
        3. Update model using teacher experience (with KL regularization)

        The teacher reward is calculated as: solved_rate * (1 - solved_rate)
        This encourages the teacher to generate problems at optimal difficulty.

        Args:
            teacher_exp: Teacher experience from teacher rollout.
            student_exps: Student experience buffers from student rollout.

        Returns:
            Combined metrics from both updates.
        """
        # Step 1: Update teacher rewards based on student performance
        # Teacher reward = solved_rate * (1 - solved_rate) per curriculum
        teacher_exp.update_teacher_rewards(
            student_buffer=student_exps,
            normalize_advantages=self.train_config.normalize_advantages,
        )

        # Step 2: Update using student experience
        student_metrics:TrainingMetrics = update(
            model=self.model,
            optimizer=self.optimizer,
            experience=student_exps,
            training_config=self.train_config,
            ref_model=self.ref_model,
        )

        # Step 3: Update using teacher experience (with updated rewards)
        teacher_metrics:TrainingMetrics = update(
            model=self.model,
            optimizer=self.optimizer,
            experience=teacher_exp,
            training_config=self.train_config,
            ref_model=self.ref_model,
        )

        # Combine metrics with prefixes
        metrics: dict[str, float] = {}
        if student_metrics is not None:
            metrics.update(student_metrics.to_dict(prefix="student/train/"))
        if teacher_metrics is not None:
            metrics.update(teacher_metrics.to_dict(prefix="teacher/train/"))

        return metrics

    def update_reference_model(self) -> None:
        """Update reference model to current model weights.

        Call this periodically during long training runs to allow
        the reference policy to track the learned policy.
        """
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def train(
        self,
        teacher_reward_fn: Optional[Callable[[SamplingOutput], dict]],
        student_reward_fn: Optional[Callable[[SamplingOutput], dict]],
        wandb_logger: Optional[wandb.log] = None,
    ) -> list[RunnerMetrics]:
        """Run the main GRPO training loop.

        Alternates between:
        1. Teacher generating curricula
        2. Student solving curricula
        3. GRPO optimization step

        Args:
            teacher_reward_fn: Reward function for teacher rollout.
            student_reward_fn: Reward function for student rollout.
            logger: Optional callback for logging metrics (e.g., wandb.log).

        Returns:
            List of RunnerMetrics objects, one per training step.
        """
        if not teacher_reward_fn:
            raise ValueError("Teacher reward function is required")
        if not student_reward_fn:
            raise ValueError("Student reward function is required")

        all_metrics: list[RunnerMetrics] = []

        for step in range(self.train_config.num_training_steps):
            logger.info(f"\n=== Step {step + 1}/{self.train_config.num_training_steps} ===")

            # Phase 1: Teacher generates curricula
            logger.info("Teacher generating curricula...")
            teacher_exp, teacher_rollout_metrics = self.rollout_teacher(
                teacher_reward_fn=teacher_reward_fn,
            )

            if teacher_exp.curriculums is None:
                logger.warning("Teacher failed to generate valid curricula. Skipping step.")
                continue

            # Phase 2: Student solves curricula
            logger.info("Student solving curricula...")
            student_exps, student_rollout_metrics = self.rollout_student(
                curriculums=teacher_exp.curriculums,
                student_reward_fn=student_reward_fn,
            )

            # Phase 3: GRPO optimization
            logger.info("Updating model weights...")
            train_metrics:dict[str, float] = self.train_step(
                teacher_exp=teacher_exp,
                student_exps=student_exps,
            )

            # Create RunnerMetrics object
            runner_metrics = RunnerMetrics(
                step=step + 1,
                teacher_mean_reward=teacher_rollout_metrics.mean_reward,
                teacher_std_reward=teacher_rollout_metrics.std_reward,
                teacher_mean_entropy=teacher_rollout_metrics.mean_entropy,
                student_mean_reward=student_rollout_metrics.mean_reward,
                student_std_reward=student_rollout_metrics.std_reward,
                student_mean_entropy=student_rollout_metrics.mean_entropy,
                student_train_loss=train_metrics.get("student/train/loss", 0.0),
                student_train_mean_kl=train_metrics.get("student/train/mean_kl", 0.0),
                student_train_policy_loss=train_metrics.get(
                    "student/train/policy_loss", 0.0
                ),
                student_train_grad_norm=train_metrics.get(
                    "student/train/grad_norm", 0.0
                ),
                teacher_train_loss=train_metrics.get("teacher/train/loss", 0.0),
                teacher_train_mean_kl=train_metrics.get("teacher/train/mean_kl", 0.0),
                teacher_train_policy_loss=train_metrics.get(
                    "teacher/train/policy_loss", 0.0
                ),
                teacher_train_grad_norm=train_metrics.get(
                    "teacher/train/grad_norm", 0.0
                ),
            )

            all_metrics.append(runner_metrics)

            # Logging
            if step % self.train_config.log_interval == 0:
                metrics_dict = runner_metrics.to_dict()
                logger.info(f"Metrics: {metrics_dict}")
                if wandb_logger:
                    wandb_logger(metrics_dict)

        return all_metrics
