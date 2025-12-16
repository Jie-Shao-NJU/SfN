from transformers import AutoModelForCausalLM
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Model paths
path_model_a = "ckpt/Qwen2.5-32B"
path_model_b = "ckpt/DeepSeek-R1-Distill-Qwen-32B"

# Load models
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(path_model_b, torch_dtype=torch.float16, device_map="cpu")

# Get state_dict
state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# Compare parameters
diffs = {}
for k in state_dict_a:
    if k in state_dict_b:
        param_a = state_dict_a[k].float()
        param_b = state_dict_b[k].float()
        diff = (param_a - param_b).norm().item()
        diffs[k] = diff
    else:
        print(f"Parameter {k} is missing in the fine-tuned model")

# Display parameters with the largest changes
sorted_diffs = sorted(diffs.items(), key=lambda x: -x[1])
print("Top 10 parameter changes (L2 norm):")
for k, v in sorted_diffs[:10]:
    print(f"{k}: Δ = {v:.4f}")

# Visualize the distribution of all changes
# import pdb; pdb.set_trace()
# 排序并准备数据
param_names = list(diffs.keys())
diff_values = list(diffs.values())

df = pd.DataFrame({
    "parameter_name": param_names,
    "l2_norm_diff": diff_values
})
df.to_csv("parameter_diffs.csv", index=False)

plt.figure(figsize=(max(12, len(diff_values) * 0.02), 5))  # 宽度自适应参数数量
plt.bar(range(len(diff_values)), diff_values, color='skyblue')

plt.title("L2 Norm of Parameter Differences")
plt.xlabel("Parameter Index")
plt.ylabel("L2 Norm Difference")

if len(param_names) <= 50:
    plt.xticks(range(len(param_names)), param_names, rotation=90, fontsize=6)
else:
    plt.xticks([])  # 太多就不显示标签了

plt.tight_layout()
plt.savefig("compare_bar.jpg", dpi=300)