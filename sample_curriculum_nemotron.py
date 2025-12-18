import time
from typing import Type

import rich
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from curriculum import BooleanConstraintsSatisfaction, Task, TeacherPrompt


def sample_curriculum_nemotron(
    task_instance: Type[Task],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_instances_to_sample: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    min_p: float = 0.0,
    max_new_tokens: int = 1024,
) -> str:
    """
    Samples a curriculum for a given task using the Nemotron-Flash-1B model.
    """
    messages: list[dict[str, str]] = TeacherPrompt().prompt(
        seed_task=task_instance.render(),
        num_instances=num_instances_to_sample,
    )

    def _flatten_messages(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(
            [f"{message['role']}\n{message['content']}" for message in messages]
        )

    text = _flatten_messages(messages)

    print(f"Input Prompt Length (chars): {len(text)}")
    print("-" * 50)
    rich.print(text)
    print("-" * 50)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Initializing generation state...")
    # Initialize CUDA graph for faster generation
    # Note: batch_size=1 as we are processing one prompt
    generation_state = model.init_cuda_graph_generation(
        max_new_tokens=max_new_tokens,
        batch_size=1,
        device="cuda",
    )

    print("Generating with CUDA graph acceleration...")
    start_time = time.time()

    # Use the CUDA graph optimized generation
    outputs: torch.Tensor = model.generate_with_cuda_graph(
        input_ids=model_inputs.input_ids,
        generation_state=generation_state,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        eos_token_id=tokenizer.eos_token_id,
    )

    elapsed_time = time.time() - start_time

    input_length = model_inputs.input_ids.shape[1]
    new_tokens: torch.Tensor = outputs[0, input_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Time taken to generate: {elapsed_time:.4f} seconds")
    return decoded_output


def load_nemotron(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load Nemotron-Flash-1B per HF guidance:
    https://huggingface.co/nvidia/Nemotron-Flash-1B
    """
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    return model, tokenizer


if __name__ == "__main__":
    model_name = "nvidia/Nemotron-Flash-1B"
    print(f"Loading model {model_name}...")
    model, tokenizer = load_nemotron(model_name)

    print("Sampling curriculum (Nemotron)...")
    output = sample_curriculum_nemotron(
        BooleanConstraintsSatisfaction(),
        model,
        tokenizer,
        max_new_tokens=4_096,
        temperature=1,
        top_p=0.95,
        top_k=20,
    )

    print("-" * 50)
    print("Sampled Output:")
    rich.print(output)
    print("-" * 50)
