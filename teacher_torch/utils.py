from typing import Optional
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch

def load_model_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool = True,
    bf16: bool = True,
    device_map: Optional[str] = "auto",
    freeze_model: bool = False,
    checkpoint_path: Optional[str] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads a pre-trained model and its tokenizer.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model.
        trust_remote_code (bool): Whether to trust remote code.
        bf16 (bool): Whether to use bfloat16 precision.
        device_map (str): The device map to use for the model.

    Returns:
        tuple: A tuple containing the model and its tokenizer.
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model: PreTrainedModel = AutoLigerKernelForCausalLM.from_pretrained(
        model_name_or_path if checkpoint_path is None else checkpoint_path,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer