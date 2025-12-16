import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM
from matplotlib import rcParams

# 解决字体问题
rcParams['font.family'] = ['Times New Roman', 'Times', 'serif']

def compute_diff_matrix(model_path_a, model_path_b, target_modules):
    model_a = AutoModelForCausalLM.from_pretrained(model_path_a, torch_dtype=torch.float16, device_map="cpu")
    model_b = AutoModelForCausalLM.from_pretrained(model_path_b, torch_dtype=torch.float16, device_map="cpu")

    state_dict_a = model_a.state_dict()
    state_dict_b = model_b.state_dict()

    pattern = re.compile(r'model\.layers\.(\d+)\.(self_attn|mlp)\.(' + '|'.join(target_modules) + r')\.(weight|bias)')
    heatmap_data = {}

    for name in state_dict_a:
        if name in state_dict_b:
            match = pattern.search(name)
            if match:
                layer_idx = int(match.group(1))
                module_name = match.group(3)
                param_a = state_dict_a[name].float()
                param_b = state_dict_b[name].float()
                diff = (param_a - param_b).norm().item()
                key = (module_name, layer_idx)
                heatmap_data[key] = heatmap_data.get(key, 0) + diff

    num_layers = max(k[1] for k in heatmap_data.keys()) + 1
    matrix = np.zeros((len(target_modules), num_layers))

    for (module, layer), diff in heatmap_data.items():
        if module in target_modules:
            module_idx = target_modules.index(module)
            matrix[module_idx, layer] = diff

    return matrix

# 模块列表
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# 差异矩阵
matrix_14b = compute_diff_matrix("ckpt/Qwen2.5-Math-7B", "ckpt/DeepSeek-R1-Distill-Qwen-7B", target_modules)
matrix_32b = compute_diff_matrix("ckpt/Llama-3.1-8B", "ckpt/DeepSeek-R1-Distill-Llama-8B", target_modules)
matrix_64b = compute_diff_matrix("ckpt/Llama-3.3-70B-Instruct", "ckpt/DeepSeek-R1-Distill-Llama-70B", target_modules)  # 替换为实际路径

# colorbar 范围统一
vmin = 0
vmax = max(matrix_14b.max(), matrix_32b.max(), matrix_64b.max())

# 设置横轴范围
max_layers = max(matrix_14b.shape[1], matrix_32b.shape[1], matrix_64b.shape[1])
xticks = np.arange(0, max_layers, step=4)

# 设置画布为三行瘦长图
fig, axes = plt.subplots(3, 1, figsize=(10, 4.5), sharex=True, sharey=True,
                         gridspec_kw={'hspace': 0.15, 'left': 0.18, 'right': 0.88, 'top': 0.95, 'bottom': 0.12})

# 绘图函数
def draw_heatmap(matrix, ax, show_colorbar=False):
    sns.heatmap(matrix, ax=ax, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                cbar=show_colorbar, xticklabels=False, yticklabels=target_modules,
                cbar_kws={"shrink": 0.6, "pad": 0.01, "ticks": np.linspace(vmin, vmax, 5)} if show_colorbar else None)
    ax.tick_params(left=True, bottom=False)
    ax.set_yticklabels(target_modules, fontsize=7)
    # ax.set_title(title, fontsize=9)

# 三个子图绘制
draw_heatmap(matrix_14b, axes[0])
draw_heatmap(matrix_32b, axes[1])
draw_heatmap(matrix_64b, axes[2], show_colorbar=True)

# 横轴设置
valid_xticks = [x for x in xticks if x < max_layers]
axes[2].set_xticks(valid_xticks)
axes[2].set_xticklabels([str(x) for x in valid_xticks], fontsize=7)
axes[2].tick_params(axis='x', labelbottom=True, bottom=True)

# colorbar 调整位置
cbar = axes[2].collections[0].colorbar
cbar.ax.tick_params(labelsize=7)
cbar.ax.set_position([0.89, 0.33, 0.015, 0.28])  # left, bottom, width, height

# 保存结果
plt.savefig("compare/heatmap_three_skinny.pdf", bbox_inches='tight', dpi=300)
