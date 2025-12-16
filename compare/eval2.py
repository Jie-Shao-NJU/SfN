import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = "qwen_s1K_tokenized_bs16_lr1e-5_epoch5_wd1e-4_20250422_231022"
path_model_a = f"ckpts/{model}"
model_a = AutoModelForCausalLM.from_pretrained(path_model_a, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(path_model_a, trust_remote_code=True)

model_a = model_a.eval()  # 切到eval模式
device = "cuda" if torch.cuda.is_available() else "cpu"

# 因为device_map="auto"，模型已经在多个GPU上被分割，因此不用再将整个模型显式移动到单个设备
# 这里可以不再使用 model_a = model_a.to(device)

# 定义附加的 prompt
prompt_suffix = (
    "Please reason step by step, and put your final answer within \\boxed{}. "
    "It's a math problem and don't output any code. "
    "Remember to put your final answer within \\boxed{}."
)

# 打开数据文件
input_file = "data/aime.jsonl"
output_file = f"compare/result/s1/generated_answers_{model}.txt"

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
            outputs = model_a.generate(**inputs, max_new_tokens=16384)
        
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
