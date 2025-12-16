import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm    # <<< 加上tqdm
from transformers import AutoModelForCausalLM

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
path_model_b = "ckpt/DeepSeek-R1-Distill-Qwen-32B"

model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(path_model_b, torch_dtype=torch.float16, device_map="cpu")

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# 支持的模块名字
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# 正则匹配
pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(" + "|".join(target_modules) + r")\.weight")

# 收集所有差异
layer_diffs = {}

for name in state_dict_a:
    if name in state_dict_b:
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            block_type = match.group(2)  # self_attn 或 mlp
            proj_name = match.group(3)   # q_proj, v_proj 等
            key = (layer_idx, block_type, proj_name)
            
            w_a = state_dict_a[name].float()
            w_b = state_dict_b[name].float()

            diff_matrix = (w_b - w_a)  # 注意方向：w_b - w_a
            layer_diffs[key] = diff_matrix

# 遍历每一层的每一个线性层，分别画图
for (layer_idx, block_type, proj_name), diff_matrix in tqdm(layer_diffs.items(), desc="Drawing heatmaps"):
    diff_flat = diff_matrix.flatten()

    # 防止超大tensor炸掉，抽样
    if diff_flat.numel() > 100000:
        idx = torch.randperm(diff_flat.numel())[:100000]
        sample = diff_flat[idx]
    else:
        sample = diff_flat

    # 自动根据99%分位数设定vmin/vmax
    vmax = torch.quantile(sample.abs(), 0.99).item()
    vmin = -vmax

    # ---- 开始画图 ----
    fig, axes = plt.subplots(2, 1, figsize=(6, 10))

    # 上面：热力图
    sns.heatmap(
        diff_matrix.cpu().numpy(),
        cmap="RdBu_r",
        center=0,
        vmin=vmin,
        vmax=vmax,
        ax=axes[0],
        cbar=True
    )
    axes[0].set_title(f"Layer {layer_idx} {block_type}.{proj_name} Weight Diff Heatmap", fontsize=12)
    axes[0].axis("off")

    # 下面：分布图
    diff_flat_np = diff_matrix.flatten().cpu().numpy()
    sns.histplot(
        diff_flat_np,
        bins=200,
        kde=False,
        ax=axes[1],
        color="blue"
    )
    axes[1].set_title("Difference Value Distribution", fontsize=12)
    axes[1].set_xlabel("Weight Difference")
    axes[1].set_ylabel("Count")

    plt.tight_layout()

    # 保存高分辨率图
    filename = f"compare/result/abs/Layer{layer_idx}_{block_type}_{proj_name}_diff_with_hist.jpg"
    plt.savefig(filename, dpi=600)
    plt.close()

print("All difference heatmaps + histograms have been saved!")
