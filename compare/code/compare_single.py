import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
path_model_b = "ckpt/DeepSeek-R1-Distill-Qwen-32B"

model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(path_model_b, torch_dtype=torch.float16, device_map="cpu")

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# 设置你想画的具体层和线性层
target_layer_idx = 10             # 指定层，比如第10层
target_proj_name = "q_proj"       # 指定线性层，比如q_proj

# 支持的模块名字
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']

# 正则：匹配任何 self_attn 或 mlp 下的线性层
pattern = re.compile(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(" + "|".join(target_modules) + r")\.weight")

# 找到对应的diff
found = False

for name in state_dict_a:
    if name in state_dict_b:
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            block_type = match.group(2)   # self_attn 或 mlp
            proj_name = match.group(3)    # 具体的线性层名字，比如 q_proj

            if layer_idx == target_layer_idx and proj_name == target_proj_name:
                w_a = state_dict_a[name].float()
                w_b = state_dict_b[name].float()
                diff_matrix = (w_b - w_a)

                diff_flat = diff_matrix.flatten()
                # 防止超大矩阵爆内存，选一部分sample
                if diff_flat.numel() > 100000:
                    idx = torch.randperm(diff_flat.numel())[:100000]
                    sample = diff_flat[idx]
                else:
                    sample = diff_flat
                # 取99%分位数
                vmax = torch.quantile(sample.abs(), 0.99).item()
                vmin = -vmax
                
                # 画一张大图，分成上下两块
                fig, axes = plt.subplots(2, 1, figsize=(6, 10))

                # ---- 上面：热力图 ----
                sns.heatmap(
                    diff_matrix.cpu().numpy(),
                    cmap="RdBu_r",
                    center=0,
                    vmin=vmin, vmax=vmax,
                    ax=axes[0],
                    cbar=True
                )
                axes[0].set_title(f"Layer {layer_idx} {block_type}.{proj_name} Weight Diff Heatmap", fontsize=12)
                axes[0].axis("off")

                # ---- 下面：分布图 ----
                diff_flat = diff_matrix.flatten().cpu().numpy()
                sns.histplot(diff_flat, bins=200, kde=False, ax=axes[1], color="blue")
                axes[1].set_title("Difference Value Distribution", fontsize=12)
                axes[1].set_xlabel("Weight Difference")
                axes[1].set_ylabel("Count")

                plt.tight_layout()

                # 保存文件
                filename = f"Layer{layer_idx}_{block_type}_{proj_name}_diff_with_hist.jpg"
                plt.savefig(filename, dpi=600)
                plt.close()
                print(f"Saved combined heatmap + distribution to {filename}")
                found = True
                break

if not found:
    print(f"Target Layer {target_layer_idx} {target_proj_name} not found!")
