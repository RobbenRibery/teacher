import jax
import os
from tunix.cli.utils import model as model_lib
from tunix.generate.sampler import (
    Sampler,
    CacheConfig,
)
from tunix.generate.base_sampler import SamplerOutput


def load_model(
    model_name: str,
    model_source: str = "huggingface",
    tokenizer_path: str | None = None,
    cache_size: int = 1024,
    intermediate_ckpt_dir: str = "/tmp/tunix_ckpt",
    model_download_path: str = "/tmp/tunix_model",
    tokenizer_type: str = "huggingface",
) -> Sampler:
    """
    Loads a model and tokenizer, returning an initialized Sampler.

    Args:
        model_name: Name of the model (e.g. "gemma-2b").
        model_source: Source of the model ("huggingface", "kaggle", "gcs").
        tokenizer_path: Path to tokenizer file. If None, will be resolved by create_model.
        cache_size: Size of the KV cache.
        intermediate_ckpt_dir: Directory for intermediate checkpoints (required for HF/Kaggle).
        model_download_path: Directory to download the model to.
        tokenizer_type: Type of tokenizer ("huggingface" or "sentencepiece").

    Returns:
        Initialized Sampler instance.
    """
    
    # Create default mesh (1 device, data parallelism)
    # Adjust axis names/shapes as needed based on available hardware
    mesh = jax.make_mesh((1, 1), ('data', 'model'))

    model_config = {
        "model_name": model_name,
        "model_source": model_source,
        "model_download_path": model_download_path,
        "intermediate_ckpt_dir": intermediate_ckpt_dir,
        "model_display": False,
        # Required for kaggle/hf if intermediate_ckpt_dir is needed
    }

    tokenizer_config = {
        "tokenizer_path": tokenizer_path,
        "tokenizer_type": tokenizer_type,
        "add_bos": True,
        "add_eos": False,
    }

    # create_model downloads/prepares the model and returns the nnx.Module + tokenizer path
    model, resolved_tokenizer_path = model_lib.create_model(
        model_config,
        tokenizer_config,
        mesh
    )

    # Ensure we use the resolved path if original was None
    if tokenizer_path is None:
        tokenizer_config["tokenizer_path"] = resolved_tokenizer_path

    tokenizer = model_lib.create_tokenizer(tokenizer_config, resolved_tokenizer_path)

    # model is an nnx.Module (likely gemma.model.Transformer)
    # We access its config to populate CacheConfig
    # Assuming model.config exists and has these attributes (verified for Gemma)
    cache_config = CacheConfig(
        cache_size=cache_size,
        num_layers=model.config.num_layers,
        num_kv_heads=model.config.num_kv_heads,
        head_dim=model.config.head_dim,
    )

    sampler = Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=cache_config,
    )

    return sampler


def generate(input_texts:str | list[str], sampler:Sampler, **kwargs) -> str | list[str]:
    out: SamplerOutput = sampler(input_strings=input_texts, **kwargs)
    if isinstance(input_texts, str):
        return out.text[0]
    else:
        return out.text
