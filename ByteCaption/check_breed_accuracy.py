#!/usr/bin/env python3
"""分析验证集的品种名准确率"""

import re
import json

# 从日志中提取Generated和Reference
log_file = "results_stage2_v2/training.log"

# 提取step 2000的评估结果
with open(log_file, 'r') as f:
    content = f.read()

# 找到step 2000的评估部分
step_2000_start = content.find("Evaluating step_2000")
if step_2000_start == -1:
    print("Step 2000 not found")
    exit(1)

# 提取到下一个评估之前
step_2500_start = content.find("Evaluating step_2500", step_2000_start)
if step_2500_start == -1:
    eval_section = content[step_2000_start:]
else:
    eval_section = content[step_2000_start:step_2500_start]

# 提取所有Generated和Reference
generated_pattern = r"Generated: the dog is a (.+?)(?:\n|$)"
reference_pattern = r"Reference: The dog is a (.+?)\."

generated = re.findall(generated_pattern, eval_section)
references = re.findall(reference_pattern, eval_section)

print(f"找到 {len(generated)} 个Generated, {len(references)} 个Reference")
print(f"数据量: {len(generated)} 样本\n")

# 提取品种名（去掉句号）
generated_breeds = [g.replace('.', '').strip().lower() for g in generated]
reference_breeds = [r.replace('.', '').strip().lower() for r in references]

# 计算准确率
correct = sum(1 for g, r in zip(generated_breeds, reference_breeds) if g == r)
accuracy = correct / len(generated_breeds) * 100 if generated_breeds else 0

print(f"品种名准确率: {correct}/{len(generated_breeds)} = {accuracy:.1f}%\n")

# 显示前20个对比
print("前20个样本对比:")
print("-" * 80)
for i in range(min(20, len(generated_breeds))):
    match = "✓" if generated_breeds[i] == reference_breeds[i] else "✗"
    print(f"{i+1:2d}. {match}  生成: {generated_breeds[i]:20s} | 真实: {reference_breeds[i]}")

# 统计最常见的错误
error_types = {}
for g, r in zip(generated_breeds, reference_breeds):
    if g != r:
        error_type = f"'{g}' vs '{r}'"
        error_types[error_type] = error_types.get(error_type, 0) + 1

if error_types:
    print(f"\n最常见的错误 (Top 10):")
    sorted_errors = sorted(error_types.items(), key=lambda x: -x[1])[:10]
    for error, count in sorted_errors:
        print(f"  {error}: {count}次")
