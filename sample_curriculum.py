import time
import torch
from typing import Type
from teacher_torch.utils import load_model_tokenizer
from curriculum import BooleanConstraintsSatisfaction, Task, TeacherPrompt

def sample_curriculum(task_instance: Type[Task], model, tokenizer) -> str:
    # Instantiate the task to get the prompt
    prompt = TeacherPrompt().render(task_instance.render())
    
    print(f"Input Prompt Length (chars): {len(prompt)}")
    print(prompt)
    # Tokenize input (automatically creates attention_mask)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    print("Generating...")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # Explicitly passing attention_mask
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Extract new tokens only
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0, input_length:]
    decoded_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    print(f"Time taken to generate: {elapsed_time:.4f} seconds")
    
    return decoded_output

if __name__ == "__main__":
    # Load model
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading model {model_name}...")
    # bf16=False for CPU safety, though utils handles fallback
    model, tokenizer = load_model_tokenizer(model_name)
    
    # Sample
    print("Sampling curriculum...")
    output = sample_curriculum(BooleanConstraintsSatisfaction(), model, tokenizer)
    
    print("-" * 50)
    print("Sampled Output:")
    print(output)
    print("-" * 50)

