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

import asyncio
import copy
from dataclasses import dataclass, field
import random
from typing import List, Optional, Dict, Any, Tuple, Callable

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizer

from teacher_torch.experience import ExperienceBuffer, Experience
from teacher_torch.rollout import rollout, RolloutMetric, RolloutConfig
from teacher_torch.sampling import SamplingOutput
from teacher_torch.utils import load_model_tokenizer
from teacher_torch.policy_gradient import update
from environment.reward import (
    reward_curriculum,
    reward_answer_group,
    AnswerReward,
)
from curriculum.prompt_template import TeacherPrompt
from curriculum.tasks import BooleanConstraintsSatisfaction
from transformers import GenerationConfig
import rich

@dataclass
class GRPOConfig:
    """Configuration for GRPO training.
    
    Attributes:
        model_name_or_path: HuggingFace model identifier or local path.
        teacher_group_size: Number of curriculum samples per seed task (G_teacher).
        student_group_size: Number of answer samples per curriculum (G_student).
        max_new_tokens: Maximum tokens to generate per completion.
        temperature: Sampling temperature (higher = more random).
        top_p: Nucleus sampling probability threshold.
        learning_rate: Optimizer learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        epochs_per_step: Number of optimization epochs per rollout batch.
        clip_eps: PPO clipping epsilon for importance ratio.
        kl_beta: KL penalty coefficient (higher = more conservative updates).
        log_interval: Steps between logging outputs.
    """
    # Model configuration
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    
    # Generation configuration
    teacher_group_size: int = 8
    student_group_size: int = 16
    max_new_tokens: int = 512
    temperature: float = 0.7
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
        
        # Regex for extracting Python code from generations
        self.format_regex = (
            r"<think>.*?</think>.*?<answer>.*?```python\n(.*?)\n```.*?</answer>"
        )

        # generation config
        self.teacher_generation_config = teacher_generation_config or GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=1,
            do_sample=True,
            num_return_sequences=config.teacher_group_size,
        )
        self.student_generation_config = student_generation_config or GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=1,
            do_sample=True,
            num_return_sequences=config.student_group_size,
        )
    
    def _build_teacher_prompt(self, seed_task: str) -> str:
        """Build formatted teacher prompt from seed task.
        
        Args:
            seed_task: High-level task description for curriculum generation.
        
        Returns:
            Formatted prompt string ready for tokenization.
        """
        prompt = self.teacher_prompt_template.prompt(
            seed_task=seed_task,
            num_instances=1,
        )
        return self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    
    def _build_student_prompt(self, curriculum: str) -> str:
        """Build formatted student prompt from curriculum.
        
        Args:
            curriculum: The curriculum/problem for the student to solve.
        
        Returns:
            Formatted prompt string ready for tokenization.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a skilled problem solver. Solve the given problem step by step.",
            },
            {
                "role": "user",
                "content": (
                    f"Solve the following problem:\n\n{curriculum}\n\n"
                    "Provide your reasoning in <think> tags and your final "
                    "answer in <answer> tags with Python code."
                ),
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    
    def rollout_teacher(
        self,
        seed_tasks: List[str],
        teacher_reward_fn: Callable[[SamplingOutput], dict],
    ) -> tuple[ExperienceBuffer, RolloutMetric]:
        """Execute teacher rollout phase.
        
        Generates multiple curricula for each seed task using the provided
        reward function.
        
        Args:
            seed_tasks: List of high-level task descriptions.
            teacher_reward_fn: Callable that takes SamplingOutput and returns
                a dict with "rewards" key containing a tensor of shape (num_samples,).
        
        Returns:
            Tuple containing:
                - experience: ExperienceBuffer with generated curricula.
                - metrics: RolloutMetric containing generation statistics.
        """
        # Build teacher prompts
        prompts = [self._build_teacher_prompt(task) for task in seed_tasks]
        
        # Create rollout config
        rollout_config = RolloutConfig(
            group_size=self.config.teacher_group_size,
            seed=self.seed,
            normalize_advantages=self.config.normalize_advantages,
        )
        
        # Generate curricula using externalized reward function
        return rollout(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            reward_fn=teacher_reward_fn,
            generation_config=self.teacher_generation_config,
            config=rollout_config,
            role="teacher",
            solved_threshold=self.config.solved_threshold,
        )
    
    def rollout_student(
        self,
        prompts: List[str],
        student_reward_fn: Callable[[SamplingOutput], dict],
    ) -> Tuple[ExperienceBuffer, RolloutMetric]:
        """Execute student rollout phase.
        
        For each valid curriculum, generates multiple answer attempts
        and computes rewards using the environment.
        
        Args:
            prompts: List of student prompts (curricula).
            student_reward_fn: Reward function for student answers.
        
        Returns:
            Tuple containing:
                - experience: ExperienceBuffer with generated answers.
                - metrics: RolloutMetric containing generation statistics.
        """
        # Format prompts for student (apply chat template)
        formatted_prompts = [self._build_student_prompt(p) for p in prompts]

        # Create rollout config
        rollout_config = RolloutConfig(
            group_size=self.config.student_group_size,
            seed=self.seed,
            normalize_advantages=self.config.normalize_advantages,
        )
        
        # Generate answers
        return rollout(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=formatted_prompts,
            reward_fn=student_reward_fn,
            generation_config=self.student_generation_config,
            config=rollout_config,
            role="student",
            solved_threshold=self.config.solved_threshold,
        )
        
    def train_step(self, teacher_exp: ExperienceBuffer, student_exp: ExperienceBuffer) -> dict[str, float]:
        """Perform a single training step with separate updates for student and teacher.
        
        Updates are performed in order:
        1. Update model using student experience (with KL regularization)
        2. Update model using teacher experience (with KL regularization)
        
        Args:
            teacher_exp: Experience buffer from teacher rollout.
            student_exp: Experience buffer from student rollout.
            
        Returns:
            Combined metrics from both updates.
        """
        # Step 1: Update using student experience
        student_metrics = update(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            experience=student_exp,
            config=self.config,
            device=self.device,
        )
        
        # Step 2: Update using teacher experience
        teacher_metrics = update(
            model=self.model,
            ref_model=self.ref_model,
            optimizer=self.optimizer,
            experience=teacher_exp,
            config=self.config,
            device=self.device,
        )
        
        # Combine metrics with prefixes
        metrics: Dict[str, float] = {}
        for key, value in student_metrics.items():
            metrics[f"student/{key}"] = value
        for key, value in teacher_metrics.items():
            metrics[f"teacher/{key}"] = value
            
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
        seed_tasks: List[str],
        num_steps: int = 100,
        teacher_reward_fn: Optional[Callable[[SamplingOutput], dict]] = None,
        student_reward_fn: Optional[Callable[[SamplingOutput], dict]] = None,
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
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
            List of metrics dictionaries, one per training step.
        """
        all_metrics: List[Dict[str, Any]] = []
        
        for step in range(num_steps):
            print(f"\n=== Step {step + 1}/{num_steps} ===")
            
            # Phase 1: Teacher generates curricula
            print("Teacher generating curricula...")
            teacher_exp, teacher_metrics = self.rollout_teacher(
                seed_tasks=seed_tasks,
                teacher_reward_fn=teacher_reward_fn,
            )
            
            # Extract generated curricula (completions from teacher)
            curriculum_generations = []
            for exp in teacher_exp.experiences:
                curriculum_generations.extend(exp.completion)
            
            # Phase 2: Student solves curricula
            print("Student solving curricula...")
            student_exp, student_metrics = self.rollout_student(
                prompts=curriculum_generations,
                student_reward_fn=student_reward_fn,
            )
            
            # Phase 3: GRPO optimization
            print("Training...")
            metrics = self.train_step(teacher_exp, student_exp)
        
            # Add step-level metrics from rollout
            metrics["step"] = step + 1
            metrics["teacher/mean_reward"] = teacher_metrics.mean_reward
            metrics["teacher/std_reward"] = teacher_metrics.std_reward
            metrics["teacher/mean_entropy"] = teacher_metrics.mean_entropy
            metrics["student/mean_reward"] = student_metrics.mean_reward
            metrics["student/std_reward"] = student_metrics.std_reward
            metrics["student/mean_entropy"] = student_metrics.mean_entropy
            
            all_metrics.append(metrics)
            
            # Logging
            if step % self.config.log_interval == 0:
                print(f"Metrics: {metrics}")
                if logger:
                    logger(metrics)
        
        return all_metrics


def main() -> None:
    """Example usage of GRPO training."""
    seed_tasks = [
        """
        Design a problem where the student must find an integer x that satisfies:
        - Mathematical constraints involving arithmetic operations
        - The solution should require multi-step reasoning
        """,
        """
        Design a problem where the student must find a value x that satisfies:
        - Constraints involving comparisons and logical conditions
        - The problem should have a unique solution
        """,
    ]
    
    config = GRPOConfig(
        model_name_or_path="Qwen/Qwen3-0.6B",
        teacher_group_size=2,
        student_group_size=2,
        learning_rate=1e-5,
        kl_beta=0.04,
    )
    
    trainer = GRPOTrainer(config)
    
    metrics = trainer.train(
        seed_tasks=seed_tasks,
        num_steps=10,
    )
    
    print("\nTraining complete!")
    print(f"Final metrics: {metrics[-1]}")


if __name__ == "__main__":
    main()
