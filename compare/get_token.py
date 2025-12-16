import re

# 读取txt文件
with open('compare/result/result-1.5B/generated_answers_Qwen2.5-Math-1.5B_mlp.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 用正则表达式提取所有 [Output Token Count: 数字]
numbers = re.findall(r'\[Output Token Count: (\d+)\]', content)
numbers = [int(num) for num in numbers]

# 计算均值
if numbers:
    mean_value = sum(numbers) / len(numbers)
    print(f'提取了 {len(numbers)} 个数，均值是 {mean_value:.2f}')
else:
    print('没有找到符合pattern的数字。')