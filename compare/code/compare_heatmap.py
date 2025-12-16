import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM

# Load models
path_model_a = "ckpt/Qwen2.5-32B"
path_model_b = "ckpt/QwQ-32B"

model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(path_model_b, torch_dtype=torch.float16, device_map="cpu")

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# 要分析的线性层名
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
heatmap_data = {}

# 正则匹配结构 model.layers.{n}.(self_attn|mlp).{proj}.(weight|bias)
pattern = re.compile(r'model\.layers\.(\d+)\.(self_attn|mlp)\.(' + '|'.join(target_modules) + r')\.(weight|bias)')

for name in state_dict_a:
    if name in state_dict_b:
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            module_name = match.group(3)  # q_proj, k_proj, etc.
            param_type = match.group(4)   # weight or bias

            param_a = state_dict_a[name].float()
            param_b = state_dict_b[name].float()
            diff = (param_a - param_b).norm().item()

            key = (layer_idx, module_name)
            heatmap_data[key] = heatmap_data.get(key, 0) + diff  # bias 和 weight 差异加在一起

# 构建热力图矩阵
num_layers = max(k[0] for k in heatmap_data.keys()) + 1
matrix = np.zeros((num_layers, len(target_modules)))

for (layer, module), diff in heatmap_data.items():
    if module in target_modules:
        module_idx = target_modules.index(module)
        matrix[layer, module_idx] = diff

# 可视化
plt.figure(figsize=(12, num_layers * 0.4))
sns.heatmap(matrix, annot=False, cmap="YlGnBu",
            xticklabels=target_modules, yticklabels=[f"Layer {i}" for i in range(num_layers)])
plt.title("Parameter Difference per Layer and Linear Module (L2 Norm, with bias)")
plt.xlabel("Module")
plt.ylabel("Layer")
plt.tight_layout()
plt.savefig("layer_finegrained_with_bias_heatmap.jpg")
plt.savefig("compare/heatmap.jpg", dpi=300)