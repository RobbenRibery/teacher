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

import copy
from dataclasses import dataclass
from typing import Optional, Any, Callable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig

from teacher_torch.experience import TeacherExperience, StudentExperienceBuffer
from teacher_torch.rollout import rollout, RolloutMetric, RolloutConfig
from teacher_torch.sampling import SamplingOutput
from teacher_torch.utils import load_model_tokenizer, configure_torch_for_training
from teacher_torch.policy_gradient import update, GRPOConfig

from environment.reward import CurriculumReward

from curriculum.prompt_template import TeacherPrompt, StudentPrompt
from curriculum.tasks import BooleanConstraintsSatisfaction

import tyro


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

    def __init__(
        self,
        config: GRPOConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[torch.device] = None,
        teacher_generation_config: Optional[GenerationConfig] = None,
        student_generation_config: Optional[GenerationConfig] = None,
        seed: int = 42,
    ) -> None:
        """Initialize the GRPO trainer.

        Args:
            config: Training configuration object.
            model: Pre-loaded model (optional, will load from config if None).
            tokenizer: Pre-loaded tokenizer (optional).
            device: Compute device (auto-detected if None).
            teacher_generation_config: Generation configuration for teacher.
            student_generation_config: Generation configuration for student.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.seed = seed

        # Load or use provided model/tokenizer
        if model is None or tokenizer is None:
            self.model, self.tokenizer = load_model_tokenizer(
                config.model_name_or_path,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        else:
            self.model = model
            self.tokenizer = tokenizer

        # Create frozen reference model for KL regularization
        self.ref_model = copy.deepcopy(self.model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # Task and prompt templates
        self.task = BooleanConstraintsSatisfaction()
        self.teacher_prompt_template = TeacherPrompt()
        self.student_prompt_template = StudentPrompt()

        # Regex for extracting Python code from generations
        self.format_regex = (
            r"<think>.*?</think>.*?<answer>.*?```python\n(.*?)\n```.*?</answer>"
        )

        # generation config
        self.teacher_generation_config = teacher_generation_config or GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=True,
            num_return_sequences=config.teacher_group_size,
        )
        self.student_generation_config = student_generation_config or GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=True,
            num_return_sequences=config.student_group_size,
        )

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
        teacher_reward_fn: Optional[Callable[[str], CurriculumReward]] = None,
        min_legit_problems: int = 0,
        max_trial_attempts: int = 10,
    ) -> tuple[TeacherExperience, RolloutMetric]:
        """Execute teacher rollout phase.

        Generates multiple curricula for a single seed task using the provided
        reward function. If min_legit_problems > 0, keeps rolling out until
        we accumulate that many correctly formatted curricula (or hit max attempts).

        Args:
            teacher_reward_fn: Callable that takes completion string and returns Reward object.
            min_legit_problems: Minimum number of correctly formatted curricula to collect.
                If 0, returns the first rollout without filtering.
            max_trial_attempts: Maximum number of rollout attempts when collecting
                correctly formatted curricula.

        Returns:
            Tuple containing:
                - experience: TeacherExperience with generated curricula.
                - metrics: RolloutMetric containing generation statistics (from last attempt).
        """
        # Build teacher prompts (single prompt)
        prompts = [self._build_teacher_prompt()]

        # Create rollout config
        rollout_config = RolloutConfig(
            group_size=self.config.teacher_group_size,
            seed=self.seed,
            normalize_advantages=self.config.normalize_advantages,
        )

        # If we don't need to filter for correct formats, just do one rollout
        if min_legit_problems <= 0:
            return rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                generation_config=self.teacher_generation_config,
                config=rollout_config,
                role="teacher",
                solved_threshold=self.config.solved_threshold,
                reward_fn=teacher_reward_fn if teacher_reward_fn is not None else None,
                logits_minibatch_size=self.config.logits_minibatch_size,
                only_check_format=teacher_reward_fn is not None,
            )
        
        # Otherwise, accumulate correctly formatted curricula
        curated_experience: Optional[TeacherExperience] = None
        attempts = 0
        metrics = None
        
        while (curated_experience is None or len(curated_experience.curriculums or []) < min_legit_problems) and attempts < max_trial_attempts:
            experience, metrics = rollout(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                generation_config=self.teacher_generation_config,
                config=rollout_config,
                role="teacher",
                solved_threshold=self.config.solved_threshold,
                reward_fn=teacher_reward_fn,
                logits_minibatch_size=self.config.logits_minibatch_size,
                only_check_format=True,  # Must check format when filtering
            )
            
            # Get correctly formatted experience
            correctly_formatted = experience.correct_formatted_experience
            if correctly_formatted is not None:
                if curated_experience is None:
                    curated_experience = correctly_formatted
                else:
                    curated_experience = curated_experience.merge(correctly_formatted)
            
            attempts += 1
        
        # Return what we've accumulated (or empty experience if nothing worked)
        if curated_experience is None:
            # Return the last failed experience
            return experience, metrics
        
        return curated_experience, metrics

    def rollout_student(
        self,
        problems: list[str],
        source_curriculums: list[str],
        student_reward_fn: Optional[Callable[[str], CurriculumReward]] = None,
    ) -> tuple[StudentExperienceBuffer, RolloutMetric]:
        """Execute student rollout phase.

        For each problem, generates multiple student answer attempts
        and computes rewards using the environment.

        Args:
            problems: List of problems for the student to solve.
            source_curriculums: List of curriculum strings that seeded each problem.
                Used to track which teacher-generated curriculum each student
                experience originated from.
            student_reward_fn: Optional reward function for student answers.

        Returns:
            Tuple containing:
                - experience: StudentExperienceBuffer with generated answers.
                - metrics: RolloutMetric containing generation statistics.
        """
        # Format prompts for student (apply chat template)
        formatted_prompts = [self._build_student_prompt(p) for p in problems]

        # Create rollout config
        rollout_config = RolloutConfig(
            group_size=self.config.student_group_size,
            seed=self.seed,
            normalize_advantages=self.config.normalize_advantages,
        )

        # Generate answers with source_curriculums for tracking
        return rollout(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=formatted_prompts,
            generation_config=self.student_generation_config,
            config=rollout_config,
            role="student",
            solved_threshold=self.config.solved_threshold,
            reward_fn=student_reward_fn,
            logits_minibatch_size=self.config.logits_minibatch_size,
            only_check_format=student_reward_fn is not None,
            source_curriculums=source_curriculums,
        )

    def train_step(
        self, teacher_exp: TeacherExperience, student_exp: StudentExperienceBuffer
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
            student_exp: Student experience buffer from student rollout.

        Returns:
            Combined metrics from both updates.
        """
        # Step 1: Update teacher rewards based on student performance
        # Teacher reward = solved_rate * (1 - solved_rate) per curriculum
        teacher_exp.update_teacher_rewards(
            student_exp,
            normalize_advantages=self.config.normalize_advantages,
        )

        # Step 2: Update using student experience
        student_metrics = update(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            experience=student_exp,
            config=self.config,
            device=self.device,
        )

        # Step 3: Update using teacher experience (with updated rewards)
        teacher_metrics = update(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            experience=teacher_exp,
            config=self.config,
            device=self.device,
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
        seed_tasks: list[str],
        num_steps: int = 100,
        teacher_reward_fn: Optional[Callable[[SamplingOutput], dict]] = None,
        student_reward_fn: Optional[Callable[[SamplingOutput], dict]] = None,
        logger: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> list[RunnerMetrics]:
        """Run the main GRPO training loop.

        Alternates between:
        1. Teacher generating curricula
        2. Student solving curricula
        3. GRPO optimization step

        Args:
            seed_tasks: List of seed task descriptions for curriculum generation.
            num_steps: Total number of training steps.
            teacher_reward_fn: Reward function for teacher rollout.
            student_reward_fn: Reward function for student rollout.
            logger: Optional callback for logging metrics (e.g., wandb.log).

        Returns:
            List of RunnerMetrics objects, one per training step.
        """
        all_metrics: list[RunnerMetrics] = []

        for step in range(num_steps):
            print(f"\n=== Step {step + 1}/{num_steps} ===")

            # Phase 1: Teacher generates curricula
            print("Teacher generating curricula...")
            teacher_exp, teacher_rollout_metrics = self.rollout_teacher(
                teacher_reward_fn=teacher_reward_fn,
            )

            # Extract generated curricula from teacher experience
            # curriculums contains the extracted python programs
            if teacher_exp.curriculums is not None:
                # Use extracted curriculums (python programs)
                curriculum_generations = teacher_exp.curriculums
                source_curriculums = teacher_exp.curriculums
            else:
                # Fallback to completions if no curriculums extracted
                curriculum_generations = teacher_exp.completions
                source_curriculums = teacher_exp.completions

            # Phase 2: Student solves curricula
            print("Student solving curricula...")
            student_exp, student_rollout_metrics = self.rollout_student(
                problems=curriculum_generations,
                source_curriculums=source_curriculums,
                student_reward_fn=student_reward_fn,
            )

            # Phase 3: GRPO optimization
            print("Training...")
            train_metrics = self.train_step(teacher_exp, student_exp)

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
            if step % self.config.log_interval == 0:
                metrics_dict = runner_metrics.to_dict()
                print(f"Metrics: {metrics_dict}")
                if logger:
                    logger(metrics_dict)

        return all_metrics


@dataclass
class CliArgs:
    """Command-line arguments for GRPO training."""

    # Model configuration
    model_name_or_path: str = "Qwen/Qwen3-0.6B"

    # Generation configuration
    teacher_group_size: int = 8
    student_group_size: int = 16
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    # Training configuration
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    epochs_per_step: int = 1

    # GRPO hyperparameters
    clip_eps: float = 0.2
    kl_beta: float = 0.04
    normalize_advantages: bool = True
    solved_threshold: float = 1.0

    # Training loop configuration
    mini_batch_size: int = 4
    logits_minibatch_size: int = 128

    # Logging configuration
    log_interval: int = 1

    # Training loop arguments
    num_steps: int = 100
    seed: int = 42

    # Seed task (can be provided as a single string or file path)
    seed_task: str = """
        Design a problem where the student must find an integer x that satisfies:
        - Mathematical constraints involving arithmetic operations
        - The solution should require multi-step reasoning
    """


def main(args: CliArgs = tyro.cli(CliArgs)) -> None:
    """Main entry point for GRPO training."""
    # Configure torch for bfloat16 training with dynamo optimizations
    configure_torch_for_training()

    # Convert CliArgs to GRPOConfig
    config = GRPOConfig(
        model_name_or_path=args.model_name_or_path,
        teacher_group_size=args.teacher_group_size,
        student_group_size=args.student_group_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        epochs_per_step=args.epochs_per_step,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        normalize_advantages=args.normalize_advantages,
        solved_threshold=args.solved_threshold,
        mini_batch_size=args.mini_batch_size,
        logits_minibatch_size=args.logits_minibatch_size,
        log_interval=args.log_interval,
    )

    # Create trainer
    trainer = GRPOTrainer(config, seed=args.seed)

    # Prepare seed tasks
    seed_tasks = [args.seed_task.strip()]

    # Run training
    metrics = trainer.train(
        seed_tasks=seed_tasks,
        num_steps=args.num_steps,
    )

    print("\nTraining complete!")
    if metrics:
        final_metrics = metrics[-1]
        print(f"Final metrics: {final_metrics.to_dict()}")


if __name__ == "__main__":
    main()
