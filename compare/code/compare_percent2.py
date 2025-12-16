import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ===== 参数控制 =====
DEBUG = False
MAX_LAYERS = 5 if DEBUG else None

path_model_a = "ckpt/Llama-3.3-70B-Instruct"
path_model_b = "ckpt/DeepSeek-R1-Distill-Llama-70B"

# ===== 加载模型 =====
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(path_model_b, torch_dtype=torch.float16, device_map="cpu")

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(" + "|".join(target_modules) + r")\.weight")

module_diff_dict = {}

# ===== 收集每层差异数据 =====
for name in tqdm(state_dict_a, desc="Collecting diffs"):
    if name in state_dict_b:
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            block_type = match.group(2)
            proj_name = match.group(3)
            module_key = f"{block_type}.{proj_name}"

            w_a = state_dict_a[name].float()
            w_b = state_dict_b[name].float()

            epsilon = 1e-8
            diff = (w_b - w_a) / (w_a.abs() + epsilon)
            diff = diff.clamp(-1.0, 1.0).flatten().cpu()

            if module_key not in module_diff_dict:
                module_diff_dict[module_key] = []
            module_diff_dict[module_key].append((layer_idx, diff))

# ===== 绘制 3D 差异分布 =====
out_dir = "compare/result/result-8B/percent-profile"
os.makedirs(out_dir, exist_ok=True)

for module_key, layer_diffs in module_diff_dict.items():
    print(f"[INFO] Drawing profile for {module_key}...")
    layer_diffs.sort(key=lambda x: x[0])
    layer_diffs = [item for item in layer_diffs if item[0] % 5 == 0]
    
    if MAX_LAYERS:
        layer_diffs = layer_diffs[:MAX_LAYERS]

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111, projection='3d')

    num_bins = 200
    bin_edges = np.linspace(-1, 1, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # colormap
    norm = mcolors.Normalize(vmin=0, vmax=max(layer_idx for layer_idx, _ in layer_diffs))
    cmap = cm.get_cmap('viridis')

    for _, (layer_idx, diff_flat) in enumerate(layer_diffs):
        # 忽略极端值
        diff_filtered = diff_flat[(diff_flat > -0.99) & (diff_flat < 0.99)]
        if diff_filtered.numel() == 0:
            continue

        hist, _ = np.histogram(diff_filtered.numpy(), bins=bin_edges)
        hist = hist / (hist.max() + 1e-6)

        xs = bin_centers
        ys = np.full_like(xs, layer_idx)
        zs = hist

        color = cmap(norm(layer_idx))
        ax.bar(xs, zs, zs=ys, zdir='y', width=0.015, alpha=0.8, color=color)

    ax.set_xlabel(r"$\Delta W$")
    ax.set_ylabel("Layer Index")
    ax.set_zlabel("Count")
    ax.set_zticks([])  # 隐藏 z 轴 tick
    ax.set_title(f"Difference Distribution Across Layers: {module_key}")
    ax.view_init(elev=30, azim=135)
    ax.grid(True)

    plt.tight_layout()
    save_path = f"{out_dir}/3d_profile_{module_key}.jpg"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Saved: {save_path}")

print("✅ All profile line plots saved.")
