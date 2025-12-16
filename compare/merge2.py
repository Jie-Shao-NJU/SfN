import re
import os
import json
import torch
import shutil
import tempfile
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
path_model_a = "Qwen2.5-Math-1.5B"
path_model_b = "DeepSeek-R1-Distill-Qwen-1.5B"

model_a = AutoModelForCausalLM.from_pretrained(f"ckpt/{path_model_a}", torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(f"ckpt/{path_model_b}", torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(f"ckpt/{path_model_a}", trust_remote_code=True)

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# 要匹配的关键词列表
rest_keys = ["q_proj", "k_proj", "v_proj"]

# 遍历 model_b 的所有参数名
for key_b in tqdm(state_dict_b.keys(), desc="Replacing selected weights"):
    if any(rest_key in key_b for rest_key in rest_keys):
        if key_b in state_dict_a:
            with torch.no_grad():
                try:
                    state_dict_a[key_b].copy_(state_dict_b[key_b])
                except:
                    print("error")
                    import pdb; pdb.set_trace()
            print(f"Replaced: {key_b}")
        else:
            print(f"Warning: {key_b} found in model_b but not in model_a")

# 加载修改后的 state_dict 到 model_a
model_a.load_state_dict(state_dict_a)

tmp_dir = "./tmp"
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
model_a.save_pretrained(tmp_dir)
del model_a, model_b
model_a = AutoModelForCausalLM.from_pretrained(
    tmp_dir,
    torch_dtype=torch.float16,
    device_map="auto",              # 强制全部加载到CPU
)
shutil.rmtree(tmp_dir)
print("All matching weights have been replaced!")

# import pdb; pdb.set_trace()
model_a = model_a.eval()  # 切到eval模式

# 定义附加的 prompt
prompt_suffix = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "It's a math problem and don't output any code. "
    "Remember to put your final answer within \\boxed{}."
)

# 打开数据文件
input_file = "data/aime.jsonl"
output_file = f"compare/result/result-1.5B/generated_answers_{path_model_a}_qkv.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in tqdm(f_in, desc="Generating answers"):
        data = json.loads(line)
        index = data["index"]
        origin_prompt = data["origin_prompt"].strip()
        
        # 拼接完整的 prompt
        full_prompt = origin_prompt + "\n\n" + prompt_suffix
        inputs = tokenizer(full_prompt, return_tensors="pt")
        
        # 把输入迁移到模型所在的设备（自动映射到多个GPU）
        # 这样，输入张量会与模型保持在相同的设备上
        inputs = {key: value.to(model_a.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model_a.generate(**inputs, max_new_tokens=8192)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_token_count = outputs.shape[-1]  # ✨ 生成的 token 数量

        # 写入到输出文件
        f_out.write("="*40 + f"\nProblem {index}\n" + "="*40 + "\n")
        f_out.write(full_prompt + "\n\n")
        f_out.write(f"\n\n[Output Token Count: {output_token_count}]\n")  # ✨ 把 token 数量也写进去
        f_out.write("Generated Answer:\n")
        f_out.write(generated_text)
        f_out.write("\n\n\n")  # 多留几个换行，方便视觉分隔
        f_out.flush()  # ✨ 关键：每次写完一题，立刻刷新到硬盘


