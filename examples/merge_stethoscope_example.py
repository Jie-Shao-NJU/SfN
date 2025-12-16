"""
Merge Stethoscope Example
Selectively merge modules from reasoning model into base model
"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def merge_modules(base_model_path, reasoning_model_path, modules_to_merge, output_path):
    """
    Merge specific modules from reasoning model into base model

    Args:
        base_model_path: Path to base model
        reasoning_model_path: Path to reasoning model
        modules_to_merge: List of module names to merge (e.g., ['o_proj', 'norm'])
        output_path: Path to save merged model
    """
    print(f"Loading base model: {base_model_path}")
    model_a = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    print(f"Loading reasoning model: {reasoning_model_path}")
    model_b = AutoModelForCausalLM.from_pretrained(
        reasoning_model_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    # Build pattern for modules to merge
    patterns = []
    for module in modules_to_merge:
        if module == "o_proj":
            patterns.append(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight")
        elif module == "norm":
            patterns.append(r"model\.norm\.weight")
        elif module == "lm_head":
            patterns.append(r"lm_head\.weight")
        elif module == "embed_tokens":
            patterns.append(r"model\.embed_tokens\.weight")

    # Find matching keys
    keys_to_replace = []
    for pattern in patterns:
        compiled_pattern = re.compile(pattern)
        keys_to_replace.extend([
            name for name in state_dict_a
            if compiled_pattern.match(name)
        ])

    print(f"\nMerging {len(keys_to_replace)} parameters...")

    # Replace weights
    for key in tqdm(keys_to_replace, desc="Replacing weights"):
        if key in state_dict_b:
            with torch.no_grad():
                state_dict_a[key].copy_(state_dict_b[key])
        else:
            print(f"Warning: {key} not found in reasoning model")

    # Load merged state dict
    model_a.load_state_dict(state_dict_a)

    # Test merged model
    print("\nTesting merged model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model_a = model_a.eval()

    test_prompt = "Can you write a short paragraph about the importance of reading books?"
    inputs = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model_a.generate(**inputs, max_new_tokens=50)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test generation: {generated_text[:200]}...")

    # Save merged model
    print(f"\nSaving merged model to {output_path}")
    model_a.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✓ Merging complete!")

if __name__ == "__main__":
    # Example: Merge only o_proj
    base_model = "Qwen/Qwen2.5-Math-1.5B"
    reasoning_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Merge o_proj + essential components
    modules = ["o_proj", "norm", "lm_head", "embed_tokens"]

    merge_modules(
        base_model_path=base_model,
        reasoning_model_path=reasoning_model,
        modules_to_merge=modules,
        output_path="./merged_model"
    )
