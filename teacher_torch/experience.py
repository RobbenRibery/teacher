from __future__ import annotations
from typing import Optional, Self
from dataclasses import dataclass, fields
import torch
from torch import Tensor
from teacher_torch.sampling import SamplingOutput


@dataclass
class Experience:
    """
    A 1/2 Dimensional representation of a collection of experiences.
    The first dimension is the batch size, the second dimension is the sequence length.
    The batch size is the number of samples in a batch (i.e. number of groups * num_samples)
    The sequence length is the length of the sequences in the batch.
    """

    prompt: str
    completions: list[str]
    role: str
    pad_token_id: int

    sequence_ids: torch.IntTensor  # (group_size, seq_len)
    sequence_attention_mask: torch.IntTensor  # (group_size, seq_len)
    action_mask: torch.BoolTensor  # (group_size, seq_len-1)
    action_log_probs: torch.FloatTensor  # (group_size, seq_len-1)

    rewards: Optional[torch.FloatTensor] = None  # (group_size, )
    only_check_format: bool = False
    solved_mask: Optional[torch.BoolTensor] = None  # (group_size, )
    advantages: Optional[torch.FloatTensor] = None  # (group_size, )
    learnability: Optional[float] = None

    # kl: Optional[torch.Tensor]  = None # (num_samples * group_size, seq_len-1) (boolean)
    # log_probs_ref: Optional[torch.Tensor] = None # (num_samples * group_size, seq_len-1)

    @property
    def correct_formatted_experience(self) -> Optional["Experience"]:
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
        return Experience(
            prompt=self.prompt,
            completions=[self.completions[i] for i in correct_indices.tolist()],
            role=self.role,
            pad_token_id=self.pad_token_id,
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

    def to(self, device: torch.device) -> Self:
        """Move all tensor fields in the experience to the specified device.

        Args:
            device: Target device to move tensors to (e.g., torch.device('cuda') or torch.device('cpu')).

        Returns:
            New Experience instance with all tensor fields moved to the device.
            Non-tensor fields (prompt, completion, role, pad_token_id) remain unchanged.
        """
        members = {}
        for field in fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor) and v is not None:
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


@dataclass
class ExperienceBuffer:
    """
    A buffer of experiences.
    Each instance represent a grouped rollout for a single instruction/prompt.
    """

    experiences: list[Experience]

    @property
    def num_unique_prompts(self) -> int:
        return len(set(exp.prompt for exp in self.experiences))

    @property
    def completions(self) -> list[str]:
        return [comp for exp in self.experiences for comp in exp.completions]

    @property
    def correct_formatted_buffer(self) -> Optional["ExperienceBuffer"]:
        correct_formatted_experiences = [
            exp.correct_formatted_experience for exp in self.experiences \
            if exp.correct_formatted_experience is not None
        ]
        if len(correct_formatted_experiences) == 0:
            return None
        return ExperienceBuffer(experiences=correct_formatted_experiences)

    @property
    def prompts(self) -> list[str]:
        return [exp.prompt for exp in self.experiences]

    def __len__(self) -> int:
        return len(self.experiences)

    def __getitem__(self, index: int) -> Experience:
        return self.experiences[index]

    def __setitem__(self, index: int, value: Experience):
        self.experiences[index] = value

    def append(self, experience: Experience):
        self.experiences.append(experience)

    def extend(self, other: ExperienceBuffer | list[Experience]):
        if isinstance(other, ExperienceBuffer):
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
            New ExperienceBuffer with all experiences moved to the device.
        """
        return ExperienceBuffer(
            experiences=[exp.to(device) for exp in self.experiences]
        )

    @classmethod
    def from_sampling_output(
        cls,
        role: str,
        sampling_output: "SamplingOutput",
        pad_token_id: int,
        action_log_probs: torch.FloatTensor,
        rewards: Optional[torch.FloatTensor] = None,
        advantages: Optional[torch.FloatTensor] = None,
        solved_mask: Optional[torch.BoolTensor] = None,
        learnability: Optional[torch.FloatTensor] = None,
        only_check_format: bool = False,
    ) -> Self:
        """
        Create an ExperienceBuffer from a SamplingOutput.

        Args:
            sampling_output: The SamplingOutput containing generated sequences.
            pad_token_id: Token ID used for padding.
            group_size: Number of samples per prompt.
            action_log_probs: Log probabilities for each action, shape (num_samples, seq_len-1).
            rewards: Scalar rewards for each sample, shape (num_samples,).
            advantages: Group-normalized advantages, shape (num_samples,).
            solved_mask: Solved mask for each sample, shape (num_samples,).
            learnability: Learnability scores per sample, shape (batch_size,).

        Returns:
            ExperienceBuffer containing one Experience per prompt group.
        """
        # prompt has length num_samples (number of groups)
        # completions has length batch_size (total number of samples)
        num_samples = sampling_output.num_samples
        batch_size = len(sampling_output.prompts)
        assert num_samples == batch_size * sampling_output.group_size, (
            f"num_samples ({num_samples}) must equal batch_size ({batch_size}) * group_size ({sampling_output.group_size})"
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

        experiences: list[Experience] = []
        for i in range(batch_size):
            # Get indices for this prompt's group
            start_idx = i * sampling_output.group_size
            end_idx = (i + 1) * sampling_output.group_size

            prompt = sampling_output.prompts[i]

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

            experience = Experience(
                prompt=prompt,
                completions=group_completions,
                role=role,
                pad_token_id=pad_token_id,
                sequence_ids=group_sequence_ids,
                sequence_attention_mask=group_sequence_attention_mask,
                action_mask=group_action_mask,
                action_log_probs=group_action_log_probs,
                rewards=group_rewards,
                only_check_format=only_check_format,
                solved_mask=group_solved_mask,
                advantages=group_advantages,
                learnability=group_learnability,
            )
            experiences.append(experience.to(device="cpu"))

        assert len(experiences) == batch_size, (
            f"Expected {batch_size} experiences, got {len(experiences)}"
        )
        return cls(experiences=experiences)
