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
    completion: list[str]
    role:str
    pad_token_id:int

    sequences: torch.IntTensor  # (num_samples * group_size, seq_len)
    attention_mask: torch.IntTensor  # (num_samples * group_size, seq_len)
    action_mask: torch.BoolTensor  # (num_samples * group_size, seq_len-1)
    action_log_probs: torch.FloatTensor  # (num_samples * group_size, seq_len-1)

    returns: Optional[torch.FloatTensor] = None   # (num_samples * group_size, )
    solved_mask: Optional[torch.BoolTensor] = None  # (num_samples * group_size, )
    advantages: Optional[torch.FloatTensor] = None  # (num_samples * group_size, )
    learnability: Optional[torch.FloatTensor] = None # (num_samples, )

    #kl: Optional[torch.Tensor]  = None # (num_samples * group_size, seq_len-1) (boolean)
    #log_probs_ref: Optional[torch.Tensor] = None # (num_samples * group_size, seq_len-1)

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
        group_size: int,
        prompts: list[str],
        completions: list[str],
        action_log_probs: torch.FloatTensor,
        rewards: Optional[torch.FloatTensor] = None,
        advantages: Optional[torch.FloatTensor] = None,
        learnability: Optional[torch.FloatTensor] = None,
        solved_threshold: float = 1.0,
    ) -> Self:
        """
        Create an ExperienceBuffer from a SamplingOutput.
        
        Args:
            sampling_output: The SamplingOutput containing generated sequences.
            pad_token_id: Token ID used for padding.
            action_log_probs: Log probabilities for each action, shape (batch_size, seq_len-1).
            rewards: Scalar rewards for each sample, shape (batch_size,).
            advantages: Group-normalized advantages, shape (batch_size,).
            group_size: Number of samples per prompt.
            learnability: Learnability scores per sample, shape (batch_size,).
            prompt: List of prompt strings, length = num_samples (number of groups)
            completions: List of completion strings, length = batch_size (total number of samples)  
            solved_threshold: Threshold for determining if a task is solved.
        
        Returns:
            ExperienceBuffer containing one Experience per prompt group.
        """
        # prompt has length num_samples (number of groups)
        # completions has length batch_size (total number of samples)
        num_samples = len(prompts)
        batch_size = len(completions)
        solved_mask: Optional[torch.BoolTensor] = None
        
        # Validate that batch_size matches num_samples * group_size
        if batch_size != num_samples * group_size:
            raise ValueError(
                f"batch_size ({batch_size}) must equal num_samples ({num_samples}) * group_size ({group_size})"
            )
        
        # Validate tensor shapes
        if action_log_probs.shape[0] != batch_size:
            raise ValueError(
                f"action_log_probs.shape[0] ({action_log_probs.shape[0]}) must equal batch_size ({batch_size})"
            )

        if rewards is not None:
            if rewards.shape[0] != batch_size:
                raise ValueError(
                    f"rewards.shape[0] ({rewards.shape[0]}) must equal batch_size ({batch_size})"
                )
            # Compute solved mask from rewards
            solved_mask:torch.BoolTensor = (rewards >= solved_threshold).bool()

        if advantages is not None:
            if advantages.shape[0] != batch_size:
                raise ValueError(
                    f"advantages.shape[0] ({advantages.shape[0]}) must equal batch_size ({batch_size})"
                )
        if learnability is not None:
            if learnability.shape[0] != batch_size:
                raise ValueError(
                    f"learnability.shape[0] ({learnability.shape[0]}) must equal batch_size ({batch_size})"
                )

        
        experiences = []
        for i in range(num_samples):
            # Get indices for this prompt's group
            start_idx = i * group_size
            end_idx = (i + 1) * group_size
            
            # Extract data for this group
            group_sequences = sampling_output.sequence_ids[start_idx:end_idx]
            group_attention_mask = sampling_output.sequence_attention_mask[start_idx:end_idx]
            group_action_mask = sampling_output.action_mask[start_idx:end_idx]
            group_action_log_probs = action_log_probs[start_idx:end_idx]

            if rewards is not None:
                group_returns = rewards[start_idx:end_idx]
            else:
                group_returns = None

            if solved_mask is not None:
                group_solved_mask = solved_mask[start_idx:end_idx]
            else:
                group_solved_mask = None

            if advantages is not None:
                group_advantages = advantages[start_idx:end_idx]
            else:
                group_advantages = None

            if learnability is not None:
                group_learnability = learnability[start_idx:end_idx]
            else:
                group_learnability = None
            
            # Extract completions for this group
            # completions is a flat list of strings with length batch_size (one per sample)
            group_completions = completions[start_idx:end_idx]
            
            experience = Experience(
                prompt=prompts[i],
                completion=group_completions,
                role=role,
                pad_token_id=pad_token_id,
                sequences=group_sequences.to("cpu"),
                attention_mask=group_attention_mask.to("cpu"),
                action_mask=group_action_mask.to("cpu"),
                action_log_probs=group_action_log_probs.to("cpu"),
                returns=group_returns.to("cpu") if group_returns is not None else None  ,
                solved_mask=group_solved_mask.to("cpu") if group_solved_mask is not None else None,
                advantages=group_advantages.to("cpu") if group_advantages is not None else None,
                learnability=group_learnability.to("cpu") if group_learnability is not None else None,
            )
            experiences.append(experience)

        assert len(experiences) == num_samples, (
            f"Expected {num_samples} experiences, got {len(experiences)}"
        )
        
        return cls(experiences=experiences)
       