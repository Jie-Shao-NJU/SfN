import torch
import types
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(path_model_a, trust_remote_code=True)

# 定义 passthrough 层
def passthrough_layer(self, hidden_states, *args, **kwargs):
    return (hidden_states,)  # tuple 返回，符合原始forward约定

# Patch 指定层
for idx in range(5, 30):
    print(f"Patching layer {idx} to skip...")
    model_a.model.layers[idx].forward = types.MethodType(passthrough_layer, model_a.model.layers[idx])

# 推理
model_a = model_a.eval().to("cuda" if torch.cuda.is_available() else "cpu")

prompt = "Can you write a short paragraph about the importance of reading books?"
inputs = tokenizer(prompt, return_tensors="pt").to(model_a.device)

with torch.no_grad():
    outputs = model_a.generate(**inputs, max_new_tokens=100)

print("Generated Text:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
