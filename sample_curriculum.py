import time
import torch
from typing import Type
from transformers import PreTrainedModel, PreTrainedTokenizer
from teacher_torch.utils import load_model_tokenizer
from curriculum import BooleanConstraintsSatisfaction, Task, TeacherPrompt
import rich


def sample_curriculum(
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
    Samples a curriculum for a given task.

    Args:
        task_instance: The task instance to sample curriculum for.
        model: The model to use for sampling.
        tokenizer: The tokenizer to use for sampling.
        num_instances_to_sample: The number of instances to sample.
        temperature: The temperature to use for sampling.
        top_p: The top_p to use for sampling.
        top_k: The top_k to use for sampling.
        min_p: The min_p to use for sampling.
        max_new_tokens: The maximum number of new tokens to generate.

    Returns:
        The sampled curriculum.
    """
    # Instantiate the task to get the prompt
    messages: list[dict[str, str]] = TeacherPrompt().prompt(
        seed_task=task_instance.render(),
        num_instances=num_instances_to_sample,
    )
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    print(f"Input Prompt Length (chars): {len(text)}")
    print("-" * 50)
    rich.print(text)
    print("-" * 50)
    # Tokenize input (automatically creates attention_mask)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    print("Generating...")
    start_time = time.time()

    with torch.no_grad():
        outputs: torch.Tensor = model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,  # Explicitly passing attention_mask
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Extract new tokens only
    input_length = model_inputs.input_ids.shape[1]
    new_tokens: torch.Tensor = outputs[0, input_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Time taken to generate: {elapsed_time:.4f} seconds")

    return decoded_output


if __name__ == "__main__":
    # Load model
    model_name = "Qwen/Qwen3-1.7B"
    print(f"Loading model {model_name}...")
    # bf16=False for CPU safety, though utils handles fallback
    model, tokenizer = load_model_tokenizer(model_name)

    # Sample
    print("Sampling curriculum...")
    output = sample_curriculum(
        BooleanConstraintsSatisfaction(),
        model,
        tokenizer,
        max_new_tokens=12_000,
        temperature=1,
    )

    print("-" * 50)
    print("Sampled Output:")
    rich.print(output)
    print("-" * 50)

    with open("output.txt", "w") as f:
        f.write(output)
