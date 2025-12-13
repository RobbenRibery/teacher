import gc
import logging
from typing import Optional, List, Any
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM
try:
    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False
try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False
import torch
import rich


def setup_logger_with_file_line(
    logger_name: str, 
    level: int = logging.INFO,
    use_colors: Optional[bool] = None
) -> logging.Logger:
    """
    Configure a logger to include filename and line number in log messages.
    Optionally uses colored output via colorlog if available.
    
    Args:
        logger_name: Name of the logger (typically __name__ from the calling module).
        level: Logging level (default: logging.INFO).
        use_colors: Whether to use colored output. If None, uses colors if colorlog is available.
    
    Returns:
        Configured logger instance.
    
    Example:
        >>> logger = setup_logger_with_file_line(__name__)
        >>> logger.info("This will show filename and line number with colors")
        >>> logger = setup_logger_with_file_line(__name__, use_colors=False)
        >>> logger.info("This will show filename and line number without colors")
    """
    logger = logging.getLogger(logger_name)
    
    # Only configure if handler doesn't already exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        
        # Determine if we should use colors
        should_use_colors = use_colors if use_colors is not None else (HAS_COLORLOG and handler.stream.isatty())
        
        if should_use_colors and HAS_COLORLOG:
            # Use colorlog for colored output
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                },
                secondary_log_colors={},
                style='%'
            )
        else:
            # Fallback to regular formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
        
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger

logger = setup_logger_with_file_line(__name__)

def release_memory(vars:List[Any]):
    for var in vars:
        del var
    gc.collect()
    torch.cuda.empty_cache()    


def configure_torch_for_training():
    """Configure torch for bfloat16 training with dynamo optimizations.
    
    This function sets up:
    1. Default dtype to bfloat16 for new tensors
    2. Dynamo scalar output capture to avoid graph breaks from .item() calls
       (required because transformers flash attention uses .item() internally)
    3. Float32 matmul precision to 'high' to silence TF32 warnings
    """
    # Ensure bfloat16 as default dtype for new tensors
    torch.set_default_dtype(torch.bfloat16)
    
    # Enable scalar output capture to avoid graph breaks from .item() calls
    # (required because transformers flash attention uses .item() internally)
    torch._dynamo.config.capture_scalar_outputs = True
    
    # Silence the TF32 warning by setting matmul precision
    # (only affects float32 ops, which we avoid, but silences the warning)
    torch.set_float32_matmul_precision('high')
    
    logger.info("Configured torch for bfloat16 training with dynamo optimizations")


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
            logger.info(f"Using LigerKernel for model {model_name_or_path}")

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.padding_side == "right":
        logger.info(f"Adjusting padding side from right to left for training")
        # training padding should be on the left
        tokenizer.padding_side = "left"

    model: PreTrainedModel = ModelClass.from_pretrained(
        model_name_or_path if checkpoint_path is None else checkpoint_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16 if bf16 and use_cuda else "auto",
        device_map=device_map,
    )
    logger.debug(f"Model loaded on {model.device}")
    logger.debug(f"Model dtype: {model.dtype}")
    logger.debug(f"Model architecture:\n{model}")
    logger.debug(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
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
