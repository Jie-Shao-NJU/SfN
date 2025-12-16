import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
path_model_a = "ckpt/Qwen2.5-32B"
path_model_b = "ckpt/DeepSeek-R1-Distill-Qwen-32B"

model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(path_model_b, torch_dtype=torch.float16, device_map="cpu")

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# 正则匹配，只找 o_proj
pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight")

# 统计覆盖了多少层
o_proj_keys = [name for name in state_dict_a if pattern.match(name)]
# rest_keys = ["lm_head.weight", "model.norm.weight"]
rest_keys = ["lm_head.weight", "model.norm.weight", "model.embed_tokens.weight", "norm"]

# import pdb; pdb.set_trace()
for key in tqdm(o_proj_keys, desc="Replacing o_proj weights"):
    if key in state_dict_b:
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            state_dict_a[key].copy_(state_dict_b[key])
            # import pdb; pdb.set_trace()
    else:
        print(f"Warning: {key} not found in model_b")

for key in tqdm(rest_keys, desc="Replacing other weights"):
    if key in state_dict_b:
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            state_dict_a[key].copy_(state_dict_b[key])
            # import pdb; pdb.set_trace()
    else:
        print(f"Warning: {key} not found in model_b")

print("All o_proj weights have been replaced!")
model_a.load_state_dict(state_dict_a)

# 检查一下模型是不是能正常推理
# 加载Tokenizer（假设是AutoTokenizer，如果不对改一下）
tokenizer = AutoTokenizer.from_pretrained(path_model_a, trust_remote_code=True)

# import pdb; pdb.set_trace()
model_a = model_a.eval()  # 切到eval模式
device = "cuda" if torch.cuda.is_available() else "cpu"
model_a = model_a.to(device)

prompt = "Every morning, Aya does a $9$ kilometer walk, and then finishes at the coffee shop. One day, she walks at $s$ kilometers per hour, and the walk takes $4$ hours, including $t$ minutes at the coffee shop. Another morning, she walks at $s+2$ kilometers per hour, and the walk takes $2$ hours and $24$ minutes, including $t$ minutes at the coffee shop. This morning, if she walks at $s+\\frac12$ kilometers per hour, how many minutes will the walk take, including the $t$ minutes at the coffee shop?\n\nPlease reason step by step, and put your final answer within \\boxed{}. It's a math problem and don't output any code. Remember to put your final answer within \\boxed{}."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model_a.generate(**inputs, max_new_tokens=64)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Text:")
print(generated_text)

# 保存模型和tokenizer
save_path = "ckpt/Qwen2.5-32B-merge-2"  # 你可以自定义一个目录
model_a.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved at {save_path}")

