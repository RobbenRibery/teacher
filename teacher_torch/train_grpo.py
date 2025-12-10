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
from torch.nn.utils import clip_grad_norm_
from transformers import PreTrainedModel, PreTrainedTokenizer

from teacher_torch.experience import Experience
from teacher_torch.sampling import sample_responses_hf
from teacher_torch.utils import load_model_tokenizer
from teacher_torch.loss import (
    grpo_loss,
    compute_group_advantages,
    sequences_log_probs,
    masked_mean,
)
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
    teacher_group_size: int = 4
    student_group_size: int = 4
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

        # generation config
        self.teacher_generation_config = teacher_generation_config or GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=1,
            do_sample=True,
        )
        self.student_generation_config = student_generation_config or GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=1,
            do_sample=True,
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
    ) -> tuple[Experience, list[str]]:
        """Execute teacher rollout phase.
        
        Generates multiple curricula for each seed task. Rewards are
        computed later after student attempts (see compute_teacher_rewards).
        
        Args:
            seed_tasks: List of high-level task descriptions.
        
        Returns:
            Tuple containing:
                - experience: Experience object (rewards/advantages initially zero).
                - curriculum_generations: List of raw generation strings.
        """
        # Build teacher prompts
        prompts = [self._build_teacher_prompt(task) for task in seed_tasks]
        
        # Generate curricula
        sampling_output = sample_responses_hf(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            generation_config=self.teacher_generation_config,
            seed=self.config.seed,
        )
        
        experience = Experience(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask[:, 1:],  # Shift to align with log_probs
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            role="teacher",
            texts=completions,
        )
        
        return experience, completions
    
    def rollout_student(
        self,
        curriculum_generations: List[str],
    ) -> Tuple[Optional[Experience], List[List[AnswerReward]]]:
        """Execute student rollout phase.
        
        For each valid curriculum, generates multiple answer attempts
        and computes rewards using the environment.
        
        Args:
            curriculum_generations: Raw curriculum generation strings from teacher.
        
        Returns:
            Tuple containing:
                - experience: Experience object with computed rewards/advantages.
                    None if no valid curricula were generated.
                - answer_rewards_grouped: List of AnswerReward lists, one per curriculum.
        """
        from environment.execution import extract_format
        
        # Extract valid curricula
        extracted_curricula: List[Optional[str]] = []
        valid_indices: List[int] = []
        
        for i, gen in enumerate(curriculum_generations):
            extracted = extract_format(gen, self.format_regex, use_last_match=True)
            extracted_curricula.append(extracted)
            if extracted:
                valid_indices.append(i)
        
        if not valid_indices:
            return None, [[]] * len(curriculum_generations)
        
        # Build student prompts for valid curricula
        prompts: List[str] = []
        curriculum_to_prompt_idx: Dict[int, int] = {}
        
        for prompt_idx, curriculum_idx in enumerate(valid_indices):
            prompts.append(self._build_student_prompt(curriculum_generations[curriculum_idx]))
            curriculum_to_prompt_idx[curriculum_idx] = prompt_idx
        
        # Generate answers
        input_ids, attention_mask, action_mask, completions = self._generate(
            prompts,
            num_samples=self.config.student_group_size,
        )
        
        # Compute behavior policy log probs
        old_log_probs = self._compute_log_probs(self.model, input_ids, attention_mask)
        
        # Compute rewards using environment
        answer_rewards_grouped: List[List[AnswerReward]] = []
        all_rewards: List[float] = []
        
        completion_idx = 0
        for curriculum_idx in range(len(curriculum_generations)):
            if curriculum_idx not in curriculum_to_prompt_idx:
                answer_rewards_grouped.append([])
                continue
            
            extracted_curriculum = extracted_curricula[curriculum_idx]
            start = completion_idx * self.config.student_group_size
            end = (completion_idx + 1) * self.config.student_group_size
            group_completions = completions[start:end]
            
            # Async reward computation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                group_rewards = loop.run_until_complete(
                    reward_answer_group(
                        extracted_curriculum=extracted_curriculum,
                        answer_generations=group_completions,
                        format_regex=self.format_regex,
                    )
                )
            finally:
                loop.close()
            
            answer_rewards_grouped.append(group_rewards)
            all_rewards.extend([r.reward for r in group_rewards])
            completion_idx += 1
        
        rewards = torch.tensor(all_rewards, dtype=torch.float32)
        
        # Compute group-relative advantages
        advantages = compute_group_advantages(
            rewards,
            group_size=self.config.student_group_size,
        )
        
        experience = Experience(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_mask=action_mask[:, 1:],
            old_log_probs=old_log_probs,
            rewards=rewards,
            advantages=advantages,
            role="student",
            texts=completions,
        )
        
        return experience, answer_rewards_grouped
    
    def compute_teacher_rewards(
        self,
        curriculum_generations: List[str],
        answer_rewards_grouped: List[List[AnswerReward]],
    ) -> Tensor:
        """Compute teacher rewards based on student performance.
        
        The teacher reward encourages curricula that are neither too easy
        nor too hard. Uses uncertainty reward: p(1-p) where p is solve rate.
        
        Args:
            curriculum_generations: Teacher's raw generation strings.
            answer_rewards_grouped: Student rewards grouped by curriculum.
        
        Returns:
            Rewards tensor. Shape: (num_curricula,)
        """
        rewards: List[float] = []
        
        for i, gen in enumerate(curriculum_generations):
            answer_rewards = (
                answer_rewards_grouped[i] 
                if i < len(answer_rewards_grouped) 
                else []
            )
            
            if not answer_rewards:
                rewards.append(0.0)
                continue
            
            curriculum_reward = reward_curriculum(
                curriculum_generation=gen,
                answer_group_rewards=answer_rewards,
                check_format_only=False,
                format_regex=self.format_regex,
            )
            rewards.append(curriculum_reward.reward)
        
        return torch.tensor(rewards, dtype=torch.float32)
    
    def train_step(
        self,
        teacher_experience: Experience,
        student_experience: Optional[Experience],
    ) -> Dict[str, float]:
        """Execute one GRPO training step.
        
        Performs gradient updates on both teacher and student experiences
        using the GRPO objective with PPO clipping and KL regularization.
        
        Args:
            teacher_experience: Experience from teacher rollout.
            student_experience: Experience from student rollout (may be None).
        
        Returns:
            Dictionary of training metrics for logging.
        """
        self.model.train()
        metrics: Dict[str, float] = {}
        
        experiences = [teacher_experience]
        if student_experience is not None:
            experiences.append(student_experience)
        
        for exp in experiences:
            if exp.input_ids.shape[0] == 0:
                continue
            
            exp = exp.to(self.model.device)
            
            for _ in range(self.config.epochs_per_step):
                self.optimizer.zero_grad()
                
                # Compute current policy log probs
                log_probs, entropy = sequences_log_probs(
                    self.model,
                    exp.input_ids,
                    exp.attention_mask,
                )
                
                # Compute reference policy log probs
                with torch.no_grad():
                    ref_log_probs, _ = sequences_log_probs(
                        self.ref_model,
                        exp.input_ids,
                        exp.attention_mask,
                    )
                
                # Compute GRPO loss
                loss, mean_kl, policy_loss = grpo_loss(
                    log_probs=log_probs,
                    old_log_probs=exp.old_log_probs,
                    ref_log_probs=ref_log_probs,
                    advantages=exp.advantages,
                    action_mask=exp.action_mask,
                    clip_eps=self.config.clip_eps,
                    kl_beta=self.config.kl_beta,
                )
                
                if not loss.isfinite():
                    print(f"Warning: Non-finite loss for {exp.role}, skipping")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                )
                
                self.optimizer.step()
                
                # Collect metrics
                prefix = f"{exp.role}/"
                metrics[f"{prefix}loss"] = loss.item()
                metrics[f"{prefix}policy_loss"] = policy_loss.item()
                metrics[f"{prefix}mean_kl"] = mean_kl.item()
                metrics[f"{prefix}grad_norm"] = (
                    grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
                )
                metrics[f"{prefix}mean_entropy"] = masked_mean(
                    entropy, exp.action_mask, dim=-1
                ).item()
                metrics[f"{prefix}mean_reward"] = exp.rewards.mean().item()
        
        return metrics
    
    def train(
        self,
        seed_task:str,
        num_steps: int = 100,
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the main GRPO training loop.
        
        Alternates between:
        1. Teacher generating curricula
        2. Student solving curricula
        3. Computing rewards
        4. GRPO optimization step
        
        Args:
            seed_tasks: List of seed task descriptions for curriculum generation.
            num_steps: Total number of training steps.
            logger: Optional callback for logging metrics (e.g., wandb.log).
        
        Returns:
            List of metrics dictionaries, one per training step.
        """
        all_metrics: List[Dict[str, Any]] = []
        
        for step in range(num_steps):
            print(f"\n=== Step {step + 1}/{num_steps} ===")
            
            # Select seed task(s) for this step
            step_seed_tasks = seed_task
            
            # Phase 1: Teacher generates curricula
            print("Teacher generating curricula...")
            teacher_exp, curriculum_generations = self.rollout_teacher(seed_tasks=step_seed_tasks)
            
            # Phase 2: Student solves curricula
            print("Student solving curricula...")
            student_exp, answer_rewards_grouped = self.rollout_student(
                curriculum_generations
            )
            
            # Phase 3: Compute teacher rewards from student performance
            print("Computing teacher rewards...")
            teacher_rewards = self.compute_teacher_rewards(
                curriculum_generations,
                answer_rewards_grouped,
            )
            teacher_exp.rewards = teacher_rewards
            teacher_exp.advantages = compute_group_advantages(
                teacher_rewards,
                group_size=self.config.teacher_group_size,
            )
            
            # Phase 4: GRPO optimization
            print("Training...")
            metrics = self.train_step(teacher_exp, student_exp)
            
            # Add step-level metrics
            metrics["step"] = step + 1
            metrics["teacher/mean_reward"] = teacher_rewards.mean().item()
            if student_exp is not None:
                metrics["student/mean_reward"] = student_exp.rewards.mean().item()
            
            all_metrics.append(metrics)
            
            # Logging
            if step % self.config.log_interval == 0:
                print(f"Metrics: {metrics}")
                if logger:
                    logger(metrics)
        
        return all_metrics
    
    def update_reference_model(self) -> None:
        """Update reference model to current model weights.
        
        Call this periodically during long training runs to allow
        the reference policy to track the learned policy.
        """
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False


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
