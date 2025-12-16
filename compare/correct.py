import json
import re

# 定义文件路径
input_file = "data/aime.jsonl"
output_file = "compare/result/result-1.5B/generated_answers_Qwen2.5-Math-1.5B_mlp.txt"

# 读取正确答案
with open(input_file, "r", encoding="utf-8") as f:
    correct_answers = {}
    for line in f:
        problem = json.loads(line)
        correct_answers[problem['index']] = problem['gold_answer']

# 解析生成答案文件
with open(output_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

generated_answers = {}
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if line == "========================================" and i + 2 < len(lines):
        # 提取问题编号
        index_line = lines[i + 1].strip()
        if index_line.startswith("Problem"):
            problem_index = int(index_line.split()[1])
            i += 3  # 跳过两行分隔符和一行编号
            
            # 收集该题内容直到下一题开始
            content_lines = []
            while i < len(lines):
                if lines[i].strip() == "========================================":
                    # 看看是不是下一个问题的起点
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith("Problem"):
                        break  # 下一个题的起点，不处理
                content_lines.append(lines[i])
                i += 1

            # 从内容中提取所有 \boxed{} 中的内容
            boxed_answers = []
            for l in content_lines:
                boxed_answers.extend(re.findall(r"\\boxed\{(.*?)\}", l))

            generated_answers[problem_index] = boxed_answers
        else:
            i += 1  # 如果不是问题编号，继续下一行
    else:
        i += 1  # 正常跳过

# 比较并输出正确答案
correct_count = 0
correct_problems = []  # 用于保存正确的题目

for idx in range(30):  # 一共30题
    generated = generated_answers.get(idx, [])
    gold = str(correct_answers.get(idx, "")).strip()

    if any(gold in ans for ans in generated):
        correct_count += 1
        correct_problems.append(idx)  # 保存正确的题目编号
        print(f"Problem {idx}: Correct! Correct: {gold}, Generated: {generated}")
    else:
        print(f"Problem {idx}: Incorrect. Correct: {gold}, Generated: {generated}")

# 输出最终结果
print(f"\nTotal Correct: {correct_count} / 30")
print(f"\nCorrect Problems: {correct_problems}")
