import torch
from transformers import AutoModelForCausalLM

# 设置模型名称
model_name = "ckpt/Qwen2.5-14B-Instruct"

# 加载模型（避免直接放到 GPU 上）
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

# 设置输出文件路径
output_file = "model_parameters_14b.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("===== 所有参数名、shape 和是否 requires_grad =====\n")
    for name, param in model.named_parameters():
        line = f"{name} | shape: {param.shape} | requires_grad: {param.requires_grad}\n"
        f.write(line)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    f.write(f"\nTotal parameters: {total_params:,}\n")
    f.write(f"Trainable parameters: {trainable_params:,}\n")

print(f"参数信息已保存至 {output_file}")
