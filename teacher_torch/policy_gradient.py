from __future__ import annotations
from dataclasses import dataclass, fields
import torch
from typing import Sequence
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from teacher_torch.experience import (
    Experience, 
    TeacherExperience,
    StudentExperience,
    StudentExperienceBuffer
)
from typing import Union
from teacher_torch.loss import sequences_log_probs, grpo_loss
from teacher_torch.configs import TrainingConfig, TrainingMetrics
from teacher_torch.utils import release_memory, zero_pad_sequences
from torch.optim import Optimizer
from torch.nn import Module

def split_experiences(experience: TeacherExperience | StudentExperienceBuffer) -> Sequence[TeacherExperience | StudentExperience]:
    """Split an Experience object into a sequence of Experience objects.
    
    Splits by sample index (deduplication is handled in TeacherExperience.merge).
    """
    if isinstance(experience, TeacherExperience):
        split_exps = []
        for idx in range(len(experience.completions)):
            split_exps.append(
                TeacherExperience(
                    prompt=experience.prompt,
                    completions=[experience.completions[idx]],
                    pad_token_id=experience.pad_token_id,
                    curriculums=[experience.curriculums[idx]] if experience.curriculums is not None else None,
                    sequence_ids=experience.sequence_ids[idx:idx+1],
                    sequence_attention_mask=experience.sequence_attention_mask[idx:idx+1],
                    action_mask=experience.action_mask[idx:idx+1],
                    action_log_probs=experience.action_log_probs[idx:idx+1],
                    action_log_probs_ref=experience.action_log_probs_ref[idx:idx+1] if experience.action_log_probs_ref is not None else None,
                    rewards=experience.rewards[idx:idx+1] if experience.rewards is not None else None,
                    solved_mask=experience.solved_mask[idx:idx+1] if experience.solved_mask is not None else None,
                    advantages=experience.advantages[idx:idx+1] if experience.advantages is not None else None,
                    learnability=experience.learnability,
                    only_check_format=experience.only_check_format,
                )
            )
        return split_exps
    elif isinstance(experience, StudentExperienceBuffer):
        return experience.experiences
    else:
        raise ValueError(f"Unsupported experience type: {type(experience)}")

def collate_experiencs(batch: Sequence[TeacherExperience | StudentExperience]) -> TeacherExperience | StudentExperience:
    """Collate a batch of Experience objects into a single Experience object.
    
    Handles variable-length sequences by padding to max length in the batch.
    """
    batch_data: dict[str, torch.Tensor | list[str] | str | None] = {}

    # We need pad_token_id from the first item
    pad_token_id = batch[0].pad_token_id
    batch_data["pad_token_id"] = pad_token_id

    # Automatically get keys from the dataclass, excluding pad_token_id which is handled above
    keys = [f.name for f in fields(batch[0]) if f.name != "pad_token_id"]

    for key in keys:
        if key in {"prompt", "learnability", "only_check_format", "source_curriculum"}:
            batch_data[key] = None
            continue

        vals = [getattr(item, key) for item in batch]
        if any(v is None for v in vals):
            batch_data[key] = None
            continue
            
        if isinstance(vals[0], torch.Tensor):
            # Handle variable-length sequences with padding
            if vals[0].ndim >= 2:  # Multi-dimensional tensors (sequence data)
                # Determine pad value based on field name
                if key == "sequence_ids":
                    pad_value = pad_token_id
                elif key in {"sequence_attention_mask"}:
                    pad_value = 0
                elif key == "action_mask":
                    pad_value = 0  # Will be cast to False for BoolTensor
                elif key in {"action_log_probs", "action_log_probs_ref"}:
                    pad_value = 0
                else:
                    pad_value = 0  # Default padding value
                
                # Pad sequences along the sequence dimension
                sequences = [v for v in vals]
                reshaped_vals = zero_pad_sequences(sequences, side="right", pad_value=pad_value)
                
                # Convert to correct dtype for action_mask
                if key == "action_mask":
                    reshaped_vals = reshaped_vals.bool()
            elif vals[0].ndim == 1:
                # For 1D tensors (rewards, advantages), just stack
                reshaped_vals = torch.stack(vals, dim=0)
                reshaped_vals = reshaped_vals.reshape(-1)
            else:
                reshaped_vals = torch.stack(vals, dim=0)
            batch_data[key] = reshaped_vals
        elif isinstance(vals[0], list):
            # Lists (prompt, completion): extend
            batch_data[key] = [x for v in vals for x in v]
        else:
            raise ValueError(f"Unsupported type: {type(vals[0])} for key {key}")

    return batch[0].__class__(**batch_data)

def update(
    model: Module,
    optimizer: Optimizer,
    experience: TeacherExperience | StudentExperienceBuffer,
    training_config: TrainingConfig,
    ref_model: Module | None = None,
) -> TrainingMetrics:
    """Update the policy model using the collected experience.

    Args:
        model: The policy model to update.
        ref_model: Reference model for KL regularization. If None, no reference model will be used.
        optimizer: Optimizer for updating model parameters.
        experience: ExperienceBuffer (multiple experiences).
        config: Training configuration.

    Returns:
        TrainingMetrics if any steps were performed, None otherwise.
    """
    model.train()

    # DataLoader
    dataloader = DataLoader(
        split_experiences(experience),
        batch_size=training_config.mini_batch_size,
        shuffle=True,
        collate_fn=collate_experiencs,
    )

    # Metrics
    total_loss = 0.0
    total_kl = 0.0
    total_policy_loss = 0.0
    total_grad_norm = 0.0
    steps = 0

    device = model.device
    for _ in range(training_config.epochs_per_step):
        for batch in dataloader:
            batch: TeacherExperience | StudentExperience = batch.to(device)
            optimizer.zero_grad()
            
            # Compute current policy log probs (without ref model if we have cached probs)
            log_probs, _, _ = sequences_log_probs(
                model=model,
                sequence_ids=batch.sequence_ids,
                sequence_attention_mask=batch.sequence_attention_mask,
                return_entropy=False,
                logits_minibatch_size=training_config.logits_minibatch_size,
                ref_model=None,
            )

            # GRPO Loss
            grpo_kwargs = {
                "log_probs": log_probs,
                "old_log_probs": batch.action_log_probs,
                "ref_log_probs": batch.action_log_probs_ref,
                "advantages": batch.advantages,
                "action_mask": batch.action_mask,
            }
            
            # Add optional GRPO hyperparameters if they exist and are not None
            if hasattr(training_config, 'clip_eps'):
                if getattr(training_config, 'clip_eps') is not None:
                    grpo_kwargs['clip_eps'] = getattr(training_config, 'clip_eps')
            if hasattr(training_config, 'kl_beta'):
                if getattr(training_config, 'kl_beta') is not None:
                    grpo_kwargs['kl_beta'] = getattr(training_config, 'kl_beta')
                else:
                    if ref_model is not None:
                        raise ValueError("kl_beta must be set if ref_model is provided")
            
            loss, mean_kl, policy_loss = grpo_loss(**grpo_kwargs)

            if not loss.isfinite():
                print(f"Loss not finite, skipping: {loss}")
                continue

            loss.backward()

            grad_norm = clip_grad_norm_(
                model.parameters(), 
                max_norm=training_config.max_grad_norm,
            )

            optimizer.step()

            total_loss += loss.item()
            total_kl += mean_kl.item()
            total_policy_loss += policy_loss.item()
            total_grad_norm += grad_norm.item()
            steps += 1

            release_memory([log_probs, loss, batch])

    return TrainingMetrics(
        loss=total_loss / steps,
        mean_kl=total_kl / steps,
        policy_loss=total_policy_loss / steps,
        grad_norm=total_grad_norm / steps,
    )
