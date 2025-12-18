from __future__ import annotations
from typing import Optional, Self
from dataclasses import dataclass, fields, field
import torch
from teacher_torch.sampling import SamplingOutput


@dataclass
class Experience:
    """
    Base experience class for rollout data.
    
    A 1/2 Dimensional representation of a collection of experiences.
    The first dimension is the batch size, the second dimension is the sequence length.
    """

    prompt: str
    completions: list[str]
    pad_token_id: int

    sequence_ids: torch.IntTensor = field(default=None)  # (group_size, seq_len)
    sequence_attention_mask: torch.IntTensor = field(default=None)  # (group_size, seq_len)
    action_mask: torch.BoolTensor = field(default=None)  # (group_size, seq_len-1)
    action_log_probs: torch.FloatTensor = field(default=None)  # (group_size, seq_len-1)

    rewards: Optional[torch.FloatTensor] = None  # (group_size, )
    solved_mask: Optional[torch.BoolTensor] = None  # (group_size, )
    advantages: Optional[torch.FloatTensor] = None  # (group_size, )
    learnability: Optional[float] = None

    def to(self, device: torch.device) -> Self:
        """Move all tensor fields in the experience to the specified device.

        Args:
            device: Target device to move tensors to (e.g., torch.device('cuda') or torch.device('cpu')).

        Returns:
            New Experience instance with all tensor fields moved to the device.
            Non-tensor fields (prompt, completion, pad_token_id) remain unchanged.
        """
        members = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor) and v is not None:
                v = v.to(device=device)
            members[f.name] = v
        return type(self)(**members)


@dataclass
class TeacherExperience(Experience):
    """
    Teacher experience that holds generated curricula.
    
    The completions field holds raw rollout outputs from teacher sampling.
    The curriculums field holds the extracted python programs that are sent to students.
    
    Only TeacherExperience supports format checking via only_check_format and
    correct_formatted_experience.
    """

    curriculums: Optional[list[str]] = None  # Extracted curriculum (python programs) from completions
    only_check_format: bool = False  # Whether this experience is only for format checking

    @property
    def correct_formatted_experience(self) -> Optional["TeacherExperience"]:
        """Get a filtered experience containing only correctly formatted samples.
        
        This is only applicable when only_check_format is True.
        Filters samples where reward >= 1.0 (indicating correct format).
        
        Returns:
            TeacherExperience with only correctly formatted samples, or None if
            only_check_format is False, rewards is None, or no correct samples exist.
        """
        if not self.only_check_format:
            return None
        if self.rewards is None:
            return None

        # Get indices where reward >= 1.0
        correct_indices = torch.where(self.rewards >= 1.0)[0]

        # If no correct samples, return None
        if len(correct_indices) == 0:
            return None

        # Filter all tensors and lists to only include correct samples
        return TeacherExperience(
            prompt=self.prompt,
            completions=[self.completions[i] for i in correct_indices.tolist()],
            pad_token_id=self.pad_token_id,
            curriculums=[self.curriculums[i] for i in correct_indices.tolist()]
            if self.curriculums is not None
            else None,
            sequence_ids=self.sequence_ids[correct_indices],
            sequence_attention_mask=self.sequence_attention_mask[correct_indices],
            action_mask=self.action_mask[correct_indices],
            action_log_probs=self.action_log_probs[correct_indices],
            rewards=self.rewards[correct_indices],
            only_check_format=self.only_check_format,
            solved_mask=self.solved_mask[correct_indices]
            if self.solved_mask is not None
            else None,
            advantages=self.advantages[correct_indices]
            if self.advantages is not None
            else None,
            learnability=self.learnability,
        )

    def merge(self, other: "TeacherExperience") -> "TeacherExperience":
        """Merge another TeacherExperience into this one.
        
        Concatenates all lists and tensors. Assumes both experiences have the same
        prompt and pad_token_id.
        
        Args:
            other: Another TeacherExperience to merge.
            
        Returns:
            New TeacherExperience with merged data.
        """
        # Concatenate lists
        merged_completions = self.completions + other.completions
        merged_curriculums = None
        if self.curriculums is not None and other.curriculums is not None:
            merged_curriculums = self.curriculums + other.curriculums
        elif self.curriculums is not None:
            merged_curriculums = self.curriculums
        elif other.curriculums is not None:
            merged_curriculums = other.curriculums
        
        # Concatenate tensors
        merged_sequence_ids = torch.cat([self.sequence_ids, other.sequence_ids], dim=0)
        merged_attention_mask = torch.cat([self.sequence_attention_mask, other.sequence_attention_mask], dim=0)
        merged_action_mask = torch.cat([self.action_mask, other.action_mask], dim=0)
        merged_action_log_probs = torch.cat([self.action_log_probs, other.action_log_probs], dim=0)
        
        merged_rewards = None
        if self.rewards is not None and other.rewards is not None:
            merged_rewards = torch.cat([self.rewards, other.rewards], dim=0)
        elif self.rewards is not None:
            merged_rewards = self.rewards
        elif other.rewards is not None:
            merged_rewards = other.rewards
            
        merged_solved_mask = None
        if self.solved_mask is not None and other.solved_mask is not None:
            merged_solved_mask = torch.cat([self.solved_mask, other.solved_mask], dim=0)
        elif self.solved_mask is not None:
            merged_solved_mask = self.solved_mask
        elif other.solved_mask is not None:
            merged_solved_mask = other.solved_mask
            
        merged_advantages = None
        if self.advantages is not None and other.advantages is not None:
            merged_advantages = torch.cat([self.advantages, other.advantages], dim=0)
        elif self.advantages is not None:
            merged_advantages = self.advantages
        elif other.advantages is not None:
            merged_advantages = other.advantages
        
        return TeacherExperience(
            prompt=self.prompt,
            completions=merged_completions,
            pad_token_id=self.pad_token_id,
            curriculums=merged_curriculums,
            sequence_ids=merged_sequence_ids,
            sequence_attention_mask=merged_attention_mask,
            action_mask=merged_action_mask,
            action_log_probs=merged_action_log_probs,
            rewards=merged_rewards,
            only_check_format=self.only_check_format,
            solved_mask=merged_solved_mask,
            advantages=merged_advantages,
            learnability=self.learnability,  # Keep first one's learnability
        )

    def update_teacher_rewards(
        self,
        student_buffer: "StudentExperienceBuffer",
        normalize_advantages: bool = True,
    ) -> None:
        """Update teacher rewards based on student performance.

        For each curriculum, finds the matching student experience
        and calculates the teacher reward using: solved_rate * (1 - solved_rate).

        This reward formula encourages the teacher to generate problems at optimal
        difficulty - not too easy (solved_rate close to 1) and not too hard
        (solved_rate close to 0).

        Args:
            student_buffer: StudentExperienceBuffer containing student rollout results.
            normalize_advantages: Whether to normalize advantages after updating rewards.
        """
        if self.curriculums is None:
            return

        # Build a mapping from source_curriculum to solved_rate
        curriculum_to_solved_rate = student_buffer.get_solved_rate_by_curriculum()

        group_size = len(self.curriculums)
        new_rewards = torch.zeros(group_size, dtype=torch.float32)

        for i, curriculum in enumerate(self.curriculums):
            if curriculum in curriculum_to_solved_rate:
                solved_rate = curriculum_to_solved_rate[curriculum]
                # Learnability reward: solved_rate * (1 - solved_rate)
                # Maximum at solved_rate = 0.5
                new_rewards[i] = solved_rate * (1.0 - solved_rate)
            else:
                # If no matching student experience, reward is 0
                new_rewards[i] = 0.0

        self.rewards = new_rewards

        # Recompute advantages based on new rewards
        if normalize_advantages and group_size > 1:
            mean_reward = new_rewards.mean()
            std_reward = new_rewards.std()
            if std_reward > 1e-8:
                self.advantages = (new_rewards - mean_reward) / std_reward
            else:
                self.advantages = new_rewards - mean_reward
        else:
            self.advantages = new_rewards - new_rewards.mean()

    @classmethod
    def from_sampling_output(
        cls,
        sampling_output: "SamplingOutput",
        pad_token_id: int,
        action_log_probs: torch.FloatTensor,
        rewards: Optional[torch.FloatTensor] = None,
        advantages: Optional[torch.FloatTensor] = None,
        solved_mask: Optional[torch.BoolTensor] = None,
        learnability: Optional[torch.FloatTensor] = None,
        only_check_format: bool = False,
        curriculums: Optional[list[str]] = None,
    ) -> Self:
        """
        Create a TeacherExperience from a SamplingOutput.
        
        Assumes batch_size = 1 (single prompt).

        Args:
            sampling_output: The SamplingOutput containing generated sequences.
            pad_token_id: Token ID used for padding.
            action_log_probs: Log probabilities for each action, shape (group_size, seq_len-1).
            rewards: Scalar rewards for each sample, shape (group_size,).
            advantages: Group-normalized advantages, shape (group_size,).
            solved_mask: Solved mask for each sample, shape (group_size,).
            learnability: Learnability score, scalar.
            only_check_format: Whether this is only for format checking.
            curriculums: Optional list of extracted curriculums (python programs).

        Returns:
            TeacherExperience with the generated curricula.
        """
        batch_size = len(sampling_output.prompts)
        if batch_size != 1:
            raise ValueError(
                f"TeacherExperience expects batch_size=1, got {batch_size}"
            )

        group_size = sampling_output.group_size
        num_samples = sampling_output.num_samples

        # Validate tensor shapes
        if action_log_probs.shape[0] != num_samples:
            raise ValueError(
                f"action_log_probs.shape[0] ({action_log_probs.shape[0]}) must equal num_samples ({num_samples})"
            )

        if rewards is not None and rewards.shape[0] != num_samples:
            raise ValueError(
                f"rewards.shape[0] ({rewards.shape[0]}) must equal num_samples ({num_samples})"
            )

        if advantages is not None and advantages.shape[0] != num_samples:
            raise ValueError(
                f"advantages.shape[0] ({advantages.shape[0]}) must equal num_samples ({num_samples})"
            )

        prompt = sampling_output.prompts[0]
        group_completions = sampling_output.completions

        learnability_value = learnability.item() if learnability is not None else None

        return cls(
            prompt=prompt,
            completions=group_completions,
            pad_token_id=pad_token_id,
            curriculums=curriculums,
            sequence_ids=sampling_output.sequence_ids,
            sequence_attention_mask=sampling_output.sequence_attention_mask,
            action_mask=sampling_output.action_mask,
            action_log_probs=action_log_probs,
            rewards=rewards,
            only_check_format=only_check_format,
            solved_mask=solved_mask,
            advantages=advantages,
            learnability=learnability_value,
        ).to(device="cpu")


@dataclass
class StudentExperience(Experience):
    """
    Student experience that tracks which curriculum it was seeded from.

    The source_curriculum field stores the curriculum (extracted python program)
    that was used to generate this student's problem prompt.
    """

    source_curriculum: Optional[str] = None  # The curriculum this student experience originated from

    @property
    def solved_rate(self) -> float:
        """Calculate the solved rate for this student experience group.

        Returns:
            Float between 0 and 1 representing the fraction of solved samples.
        """
        if self.solved_mask is None:
            return 0.0
        return self.solved_mask.float().mean().item()


@dataclass
class StudentExperienceBuffer:
    """
    A buffer of student experiences.
    Each instance represents a grouped rollout for a single curriculum/problem.
    """

    experiences: list[StudentExperience] = field(default_factory=list)

    @property
    def num_unique_prompts(self) -> int:
        return len(set(exp.prompt for exp in self.experiences))

    @property
    def completions(self) -> list[str]:
        return [comp for exp in self.experiences for comp in exp.completions]

    @property
    def prompts(self) -> list[str]:
        return [exp.prompt for exp in self.experiences]

    def get_solved_rate_by_curriculum(self) -> dict[str, float]:
        """Get the solved rate for each source curriculum.

        Returns:
            Dictionary mapping source_curriculum to solved_rate.
        """
        curriculum_to_solved_rate: dict[str, float] = {}

        for exp in self.experiences:
            if exp.source_curriculum is not None:
                curriculum_to_solved_rate[exp.source_curriculum] = exp.solved_rate

        return curriculum_to_solved_rate

    def __len__(self) -> int:
        return len(self.experiences)

    def __getitem__(self, index: int) -> StudentExperience:
        return self.experiences[index]

    def __setitem__(self, index: int, value: StudentExperience):
        self.experiences[index] = value

    def append(self, experience: StudentExperience):
        self.experiences.append(experience)

    def extend(self, other: "StudentExperienceBuffer | list[StudentExperience]"):
        if isinstance(other, StudentExperienceBuffer):
            self.experiences.extend(other.experiences)
        else:
            self.experiences.extend(other)

    def __iter__(self):
        return iter(self.experiences)

    def to(self, device: torch.device) -> Self:
        """Move all experiences in the buffer to the specified device.

        Args:
            device: Target device to move tensors to.

        Returns:
            New StudentExperienceBuffer with all experiences moved to the device.
        """
        return StudentExperienceBuffer(
            experiences=[exp.to(device) for exp in self.experiences]
        )

    @classmethod
    def from_sampling_output(
        cls,
        sampling_output: "SamplingOutput",
        pad_token_id: int,
        action_log_probs: torch.FloatTensor,
        source_curriculums: list[str],
        rewards: Optional[torch.FloatTensor] = None,
        advantages: Optional[torch.FloatTensor] = None,
        solved_mask: Optional[torch.BoolTensor] = None,
        learnability: Optional[torch.FloatTensor] = None,
    ) -> Self:
        """
        Create a StudentExperienceBuffer from a SamplingOutput.

        Args:
            sampling_output: The SamplingOutput containing generated sequences.
            pad_token_id: Token ID used for padding.
            action_log_probs: Log probabilities for each action, shape (num_samples, seq_len-1).
            source_curriculums: List of curriculum strings that seeded each prompt.
                Length should equal batch_size (number of unique prompts).
            rewards: Scalar rewards for each sample, shape (num_samples,).
            advantages: Group-normalized advantages, shape (num_samples,).
            solved_mask: Solved mask for each sample, shape (num_samples,).
            learnability: Learnability scores per sample, shape (batch_size,).

        Returns:
            StudentExperienceBuffer containing one StudentExperience per prompt group.
        """
        num_samples = sampling_output.num_samples
        batch_size = len(sampling_output.prompts)
        assert num_samples == batch_size * sampling_output.group_size, (
            f"num_samples ({num_samples}) must equal batch_size ({batch_size}) * group_size ({sampling_output.group_size})"
        )

        # Validate source_curriculums length
        if len(source_curriculums) != batch_size:
            raise ValueError(
                f"source_curriculums length ({len(source_curriculums)}) must equal batch_size ({batch_size})"
            )

        # Validate tensor shapes
        if action_log_probs.shape[0] != num_samples:
            raise ValueError(
                f"action_log_probs.shape[0] ({action_log_probs.shape[0]}) must equal num_samples ({num_samples})"
            )

        if rewards is not None:
            if rewards.shape[0] != num_samples:
                raise ValueError(
                    f"rewards.shape[0] ({rewards.shape[0]}) must equal num_samples ({num_samples})"
                )

        if advantages is not None:
            if advantages.shape[0] != num_samples:
                raise ValueError(
                    f"advantages.shape[0] ({advantages.shape[0]}) must equal num_samples ({num_samples})"
                )
        if learnability is not None:
            if learnability.shape[0] != num_samples:
                raise ValueError(
                    f"learnability.shape[0] ({learnability.shape[0]}) must equal num_samples ({num_samples})"
                )

        experiences: list[StudentExperience] = []
        for i in range(batch_size):
            # Get indices for this prompt's group
            start_idx = i * sampling_output.group_size
            end_idx = (i + 1) * sampling_output.group_size

            prompt = sampling_output.prompts[i]
            source_curriculum = source_curriculums[i]

            # Extract data for this group of samples
            group_sequence_ids = sampling_output.sequence_ids[start_idx:end_idx]
            group_sequence_attention_mask = sampling_output.sequence_attention_mask[
                start_idx:end_idx
            ]
            group_action_mask = sampling_output.action_mask[start_idx:end_idx]
            group_action_log_probs = action_log_probs[start_idx:end_idx]
            group_completions = sampling_output.completions[start_idx:end_idx]

            if rewards is not None:
                group_rewards = rewards[start_idx:end_idx]
            else:
                group_rewards = None

            if solved_mask is not None:
                group_solved_mask = solved_mask[start_idx:end_idx]
            else:
                group_solved_mask = None

            if advantages is not None:
                group_advantages = advantages[start_idx:end_idx]
            else:
                group_advantages = None

            if learnability is not None:
                group_learnability = learnability[i].item()
            else:
                group_learnability = None

            experience = StudentExperience(
                prompt=prompt,
                completions=group_completions,
                pad_token_id=pad_token_id,
                source_curriculum=source_curriculum,
                sequence_ids=group_sequence_ids,
                sequence_attention_mask=group_sequence_attention_mask,
                action_mask=group_action_mask,
                action_log_probs=group_action_log_probs,
                rewards=group_rewards,
                solved_mask=group_solved_mask,
                advantages=group_advantages,
                learnability=group_learnability,
            )
            experiences.append(experience.to(device="cpu"))

        assert len(experiences) == batch_size, (
            f"Expected {batch_size} experiences, got {len(experiences)}"
        )
        return cls(experiences=experiences)
