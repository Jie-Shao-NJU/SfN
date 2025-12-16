import re
import os
import json
import torch
import shutil
import tempfile
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

# 加载模型
path_model_a = "Qwen2.5-32B"
path_model_b = "DeepSeek-R1-Distill-Qwen-32B"

model_a = AutoModelForCausalLM.from_pretrained(f"ckpt/{path_model_a}", torch_dtype=torch.float16, device_map="cpu")
model_b = AutoModelForCausalLM.from_pretrained(f"ckpt/{path_model_b}", torch_dtype=torch.float16, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(f"ckpt/{path_model_a}", trust_remote_code=True)

state_dict_a = model_a.state_dict()
state_dict_b = model_b.state_dict()

# 不希望被替换的关键词列表
no_replace_keys = ["o_proj", "post_attention_layernorm", "lm_head", "embed_tokens"]

# 遍历 model_b 的所有参数名
for key_b in tqdm(state_dict_b.keys(), desc="Replacing selected weights (inverted logic)"):
    if not any(no_key in key_b for no_key in no_replace_keys):  # 👈 注意这里是 not any
        if key_b in state_dict_a:
            with torch.no_grad():
                try:
                    state_dict_a[key_b].copy_(state_dict_b[key_b])
                except Exception as e:
                    print(f"Error copying {key_b}: {e}")
                    import pdb; pdb.set_trace()
            print(f"Replaced: {key_b}")
        else:
            print(f"Warning: {key_b} found in model_b but not in model_a")
    else:
        print(f"Skipped (protected): {key_b}")

# 加载修改后的 state_dict 到 model_a
model_a.load_state_dict(state_dict_a)

tmp_dir = "./tmp2"
if os.path.exists(tmp_dir):
    shutil.rmtree(tmp_dir)
model_a.save_pretrained(tmp_dir)
del model_a, model_b
model_a = AutoModelForCausalLM.from_pretrained(
    tmp_dir,
    torch_dtype=torch.float16,
    device_map="auto",
)
shutil.rmtree(tmp_dir)
print("All selected weights have been replaced!")
model_a = model_a.eval()

# 定义附加的 prompt
prompt_suffix = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "It's a math problem and don't output any code. "
    "Remember to put your final answer within \\boxed{}."
)

# 打开数据文件
input_file = "data/aime.jsonl"
output_file = f"compare/result/result-32B/generated_answers_{path_model_a}_test6_wo_oAnormAheadAembed.txt"

with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
    for line in tqdm(f_in, desc="Generating answers"):
        data = json.loads(line)
        index = data["index"]
        origin_prompt = data["origin_prompt"].strip()
        
        # 拼接完整的 prompt
        full_prompt = origin_prompt + "\n\n" + prompt_suffix
        inputs = tokenizer(full_prompt, return_tensors="pt")
        inputs = {key: value.to(model_a.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model_a.generate(**inputs, max_new_tokens=16384)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_token_count = outputs.shape[-1]

        f_out.write("="*40 + f"\nProblem {index}\n" + "="*40 + "\n")
        f_out.write(full_prompt + "\n\n")
        f_out.write(f"\n\n[Output Token Count: {output_token_count}]\n")
        f_out.write("Generated Answer:\n")
        f_out.write(generated_text)
        f_out.write("\n\n\n")
        f_out.flush()
