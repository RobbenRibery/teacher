import gc
from typing import Optional, List, Any
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
import torch
import rich

def release_memory(vars:List[Any]):
    for var in vars:
        del var
    gc.collect()
    torch.cuda.empty_cache()    

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
    use_cuda = torch.cuda.is_available()

    # Determine attention implementation and model class
    attn_implementation = "eager"
    ModelClass = AutoModelForCausalLM

    if use_cuda:
        major_version, _ = torch.cuda.get_device_capability()
        if major_version >= 8:
            attn_implementation = "flash_attention_2"
        
        if HAS_LIGER:
            ModelClass = AutoLigerKernelForCausalLM

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model: PreTrainedModel = ModelClass.from_pretrained(
        model_name_or_path if checkpoint_path is None else checkpoint_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16 if bf16 and use_cuda else "auto",
        device_map=device_map,
    )
    print(f"Model loaded on {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Model architecture:\n{model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    if freeze_model:
        for param in model.parameters():
            param.requires_grad = False

    return model, tokenizer


def expand_prompts_for_group(prompts: list[str], group_size: int) -> list[str]:
    """
    Expand prompts by repeating each prompt group_size times.
    
    Useful when you want to manually control batching instead of using
    num_return_sequences in generation_config.
    
    Args:
        prompts: Original list of prompts.
        group_size: Number of times to repeat each prompt.
    
    Returns:
        Expanded list where each prompt appears group_size times consecutively.
    
    Example:
        >>> expand_prompts_for_group(["a", "b"], group_size=3)
        ["a", "a", "a", "b", "b", "b"]
    """
    return [p for p in prompts for _ in range(group_size)]


if __name__ == "__main__":
    model, tokenizer = load_model_tokenizer("Qwen/Qwen3-0.6B")
    rich.print(model)
    rich.print(tokenizer)

    encoding = tokenizer.encode("Hello, world!", return_tensors="pt").to(model.device)
    rich.print(encoding)

    output = model.generate(encoding, max_new_tokens=10, do_sample=False)
    rich.print(output)
    rich.print(tokenizer.decode(output[0], skip_special_tokens=True))
