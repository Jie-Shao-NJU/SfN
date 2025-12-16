"""
Delta Stethoscope Example
Compute and visualize weight differences between base and reasoning models
"""

import torch
from transformers import AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt

def compute_weight_differences(base_model_path, reasoning_model_path):
    """
    Compute L2 norm of weight differences for each module
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

    # Compute differences for each module type
    module_diffs = {
        'q_proj': [],
        'k_proj': [],
        'v_proj': [],
        'o_proj': [],
        'up_proj': [],
        'gate_proj': [],
        'down_proj': []
    }

    for name, param_a in model_a.named_parameters():
        if name in model_b.state_dict():
            param_b = model_b.state_dict()[name]
            diff = torch.norm(param_a - param_b).item()

            for module_type in module_diffs.keys():
                if module_type in name:
                    module_diffs[module_type].append(diff)
                    break

    return module_diffs

def plot_weight_differences(module_diffs, save_path="weight_diff.png"):
    """
    Plot weight differences across modules
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    module_names = list(module_diffs.keys())
    avg_diffs = [np.mean(diffs) if diffs else 0 for diffs in module_diffs.values()]

    bars = ax.bar(module_names, avg_diffs)

    # Highlight o_proj
    bars[3].set_color('red')

    ax.set_ylabel('Average L2 Norm of Weight Difference')
    ax.set_xlabel('Module Type')
    ax.set_title('Weight Differences Between Base and Reasoning Models')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # Example usage
    base_model = "Qwen/Qwen2.5-Math-1.5B"
    reasoning_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    print("Computing weight differences...")
    diffs = compute_weight_differences(base_model, reasoning_model)

    print("\nAverage L2 norm differences:")
    for module, values in diffs.items():
        if values:
            print(f"  {module}: {np.mean(values):.4f}")

    print("\nGenerating visualization...")
    plot_weight_differences(diffs)
