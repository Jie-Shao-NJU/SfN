import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
state_dict_a = model_a.state_dict()

# 匹配所有 self_attn.o_proj.weight
pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight")
# pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight")
o_proj_keys = [name for name in state_dict_a if pattern.match(name)]
import pdb; pdb.set_trace()
o_proj_keys = o_proj_keys[5:30]

# 重新初始化 o_proj：保留原始 mean/std
print("Reinitializing o_proj weights with original mean/std...")
for key in tqdm(o_proj_keys, desc="Reinitializing"):
    # import pdb; pdb.set_trace()
    weight = state_dict_a[key]
    with torch.no_grad():
        # mean = weight.mean()
        # std = weight.std()
        torch.nn.init.zeros_(weight)
        # import pdb; pdb.set_trace()
        # torch.nn.init.normal_(weight, mean=0, std=0.02)

# 加载新 state_dict
model_a.load_state_dict(state_dict_a)

# 推理准备
tokenizer = AutoTokenizer.from_pretrained(path_model_a, trust_remote_code=True)
model_a = model_a.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_a = model_a.to(device)

# 示例输入
prompt = "Can you write a short paragraph about the importance of reading books?"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 文本生成
with torch.no_grad():
    outputs = model_a.generate(**inputs, max_new_tokens=100)

# 解码结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)
