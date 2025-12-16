import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
state_dict_a = model_a.state_dict()

# 初始化目标层范围
start, end = 63, 64   # 可以自定义层范围

# 正则模式匹配 MLP 中的三个投影子层
# mlp_patterns = [
#     re.compile(rf"model\.layers\.({i})\.mlp\.gate_proj\.weight") for i in range(start, end)
# ] + [
#     re.compile(rf"model\.layers\.({i})\.mlp\.up_proj\.weight") for i in range(start, end)
# ] + [
#     re.compile(rf"model\.layers\.({i})\.mlp\.down_proj\.weight") for i in range(start, end)
# ]
mlp_patterns = [
    re.compile(rf"model\.layers\.({i})\.self_attn\.k_proj\.weight") for i in range(start, end)
] + [
    re.compile(rf"model\.layers\.({i})\.self_attn\.k_proj\.bias") for i in range(start, end)
]

# 收集目标参数
mlp_keys = []
for name in state_dict_a:
    if any(p.match(name) for p in mlp_patterns):
        mlp_keys.append(name)

import pdb; pdb.set_trace()
# 重新初始化所有匹配的权重
print(f"Reinitializing MLP weights from layer {start} to {end}...")
for key in tqdm(mlp_keys, desc="Reinitializing"):
    weight = state_dict_a[key]
    with torch.no_grad():
        # torch.nn.init.normal_(weight, mean=0, std=0.02)
        # 或者使用 zeros_：
        torch.nn.init.zeros_(weight)

# 加载新 state_dict
model_a.load_state_dict(state_dict_a)

# 推理准备
tokenizer = AutoTokenizer.from_pretrained(path_model_a, trust_remote_code=True)
model_a = model_a.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# 示例输入
prompt = "Can you write a short paragraph about the importance of reading books?"
inputs = tokenizer(prompt, return_tensors="pt").to(model_a.device)

# 文本生成
with torch.no_grad():
    outputs = model_a.generate(**inputs, max_new_tokens=100)

# 解码结果
print("Generated Text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
