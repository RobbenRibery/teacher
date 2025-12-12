from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.functional import pad

from teacher_torch.experience import Experience, ExperienceBuffer
from teacher_torch.loss import sequences_log_probs, grpo_loss

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.nn import Module
    from teacher_torch.train_grpo import GRPOConfig

def zero_pad_sequences(sequences: List[torch.Tensor], side: str = "left", value: int = 0) -> torch.Tensor:
    """ """
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(pad(seq, padding, value=value))
    return torch.stack(padded_sequences, dim=0)

def collate_experience(batch: List[Experience]) -> Experience:
    """Collate a batch of Experience objects into a single Experience object."""
    # keys to process
    keys = [
        "sequences", "attention_mask", "action_mask", "action_log_probs",
        "returns", "solved_mask", "advantages", "learnability",
        "prompt", "completion", "role", "kl", "log_probs_ref"
    ]
    
    batch_data = {}
    
    # We need pad_token_id from the first item
    pad_token_id = batch[0].pad_token_id
    
    for key in keys:
        vals = [getattr(item, key) for item in batch]
        
        # Handle None (if any is None, the result is None)
        if any(v is None for v in vals):
            batch_data[key] = None
            continue
            
        if isinstance(vals[0], torch.Tensor):
            # Tensors need special handling based on dimensions
            # All input tensors are (G, ...). We want to output (B*G, ...)
            # Flatten the batch of groups into a single list of items
            
            flat_vals = []
            for v in vals:
                # v is (G, ...)
                # Split it into G items to pad them individually if needed?
                # Actually, sequences are (G, L). We want to pad all BG sequences to max L.
                # So we flatten to list of (1, L) or just list of (L) tensors?
                # v[i] is (L) or (L-1) etc.
                for i in range(v.size(0)):
                    flat_vals.append(v[i])
            
            if key in ["sequences", "attention_mask", "action_mask", "action_log_probs", "log_probs_ref", "kl"]:
                # These have variable sequence lengths. Pad them.
                padding_value = 0
                if key == "sequences":
                    padding_value = pad_token_id
                
                batch_data[key] = zero_pad_sequences(flat_vals, side="left", value=padding_value)
            else:
                # returns, solved_mask, advantages, learnability are (G,) or (G, 1)
                # Just stack/cat. flattening flat_vals gives list of scalars/1-element tensors
                # torch.stack(flat_vals) works if they are tensors.
                # If v[i] is scalar tensor, stack makes (BG,).
                batch_data[key] = torch.stack(flat_vals, dim=0)
                 
        elif isinstance(vals[0], list):
            # Lists (prompt, completion): extend
            new_list = []
            for v in vals:
                new_list.extend(v)
            batch_data[key] = new_list
        else:
            # Scalar or string (role, pad_token_id)
            # Take the first one
            batch_data[key] = vals[0]
            
    return Experience(**batch_data)

def update(
    model: Module,
    ref_model: Module,
    optimizer: Optimizer,
    experience: ExperienceBuffer,
    config: GRPOConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Update the policy model using the collected experience."""
    model.train()
    
    # DataLoader
    dataloader = DataLoader(
        experience,
        batch_size=config.mini_batch_size,
        shuffle=True,
        collate_fn=collate_experience,
    )
    
    # Metrics
    total_loss = 0.0
    total_kl = 0.0
    total_policy_loss = 0.0
    total_grad_norm = 0.0
    steps = 0
    
    for _ in range(config.epochs_per_step):
        for batch in dataloader:
            batch: Experience = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass - current policy
            # batch.sequences: (B*G, L)
            log_probs, _ = sequences_log_probs(
                model=model,
                sequence_ids=batch.sequences,
                attention_mask=batch.attention_mask,
                return_entropy=False,
                logits_minibatch_size=config.logits_minibatch_size,
            )
            
            # Reference policy (compute if needed)
            log_probs_ref = batch.log_probs_ref
            if log_probs_ref is None and config.kl_beta > 0:
                with torch.no_grad():
                    log_probs_ref, _ = sequences_log_probs(
                        model=ref_model,
                        sequence_ids=batch.sequences,
                        attention_mask=batch.attention_mask,
                        return_entropy=False,
                        logits_minibatch_size=config.logits_minibatch_size,
                    )
            
            # GRPO Loss
            loss, mean_kl, policy_loss = grpo_loss(
                log_probs=log_probs,
                old_log_probs=batch.action_log_probs,
                ref_log_probs=log_probs_ref,
                advantages=batch.advantages,
                action_mask=batch.action_mask,
                clip_eps=config.clip_eps,
                kl_beta=config.kl_beta,
            )
            
            if not loss.isfinite():
                print(f"Loss not finite, skipping: {loss}")
                continue
            
            loss.backward()
            
            grad_norm = clip_grad_norm_(
                model.parameters(), 
                max_norm=config.max_grad_norm
            )
            
            optimizer.step()
            
            total_loss += loss.item()
            total_kl += mean_kl.item()
            total_policy_loss += policy_loss.item()
            total_grad_norm += grad_norm.item()
            steps += 1
            
            del log_probs, log_probs_ref, loss

    if steps > 0:
        return {
            "train/loss": total_loss / steps,
            "train/mean_kl": total_kl / steps,
            "train/policy_loss": total_policy_loss / steps,
            "train/grad_norm": total_grad_norm / steps,
        }
    return {}
