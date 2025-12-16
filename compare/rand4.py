import torch
import types
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(path_model_a, trust_remote_code=True)

# 用来记录每一层的 L2 范数
residual_1_norms = []
attn_output_norms = []
residual_2_norms = []
mlp_output_norms = []

# 定义一个 hook 版本的 forward
def hooked_forward(self, hidden_states, *args, **kwargs):
    residual_1 = hidden_states
    hidden_states_norm1 = torch.norm(residual_1.float(), p=2).item()

    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states,
        **kwargs,
    )
    attn_output = hidden_states
    attn_output_norm = torch.norm(attn_output.float(), p=2).item()

    hidden_states = residual_1 + hidden_states
    residual_2 = hidden_states
    hidden_states_norm2 = torch.norm(residual_2.float(), p=2).item()

    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    mlp_output = hidden_states
    mlp_output_norm = torch.norm(mlp_output.float(), p=2).item()

    hidden_states = residual_2 + hidden_states

    # 记录
    residual_1_norms.append(hidden_states_norm1)
    attn_output_norms.append(attn_output_norm)
    residual_2_norms.append(hidden_states_norm2)
    mlp_output_norms.append(mlp_output_norm)

    return (hidden_states,)

# Patch 所有层
for idx, layer in enumerate(model_a.model.layers):
    model_a.model.layers[idx].forward = types.MethodType(hooked_forward, model_a.model.layers[idx])

# 推理
model_a = model_a.eval().to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Can you write a short paragraph about the importance of reading books?"
inputs = tokenizer(prompt, return_tensors="pt").to(model_a.device)

with torch.no_grad():
    outputs = model_a.generate(**inputs, max_new_tokens=1)

print("Generated Text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


plt.figure(figsize=(12, 7))

# 设置每条线的风格
plt.plot(residual_1_norms, label="Residual 1 Norm", linewidth=2.5, marker='o', markersize=6)
plt.plot(attn_output_norms, label="Attention Output Norm", linewidth=2.5, marker='s', markersize=6)
plt.plot(residual_2_norms, label="Residual 2 Norm", linewidth=2.5, marker='^', markersize=6)
plt.plot(mlp_output_norms, label="MLP Output Norm", linewidth=2.5, marker='d', markersize=6)

plt.xlabel("Layer", fontsize=14)
plt.ylabel("L2 Norm", fontsize=14)
plt.title("Layer-wise L2 Norms in Qwen2DecoderLayer", fontsize=16)

plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 更大的x/y轴刻度
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 保存图片
plt.tight_layout()
plt.savefig("feat_visual.png", dpi=300)  # 提高清晰度
plt.show()