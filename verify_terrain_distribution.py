#!/usr/bin/env python3
"""
验证地形分布修复：
1. 难度级别分布（行）：80% 难度0-1，15% 难度2，5% 视觉探索，跳过3-4
2. 地形类型分布（列）：按 terrain_proportions 配置随机分配，不是均匀分配
"""

import torch
import numpy as np

# 模拟配置
num_envs = 512
num_rows = 10  # 难度级别数
num_cols = 8   # 地形类型数

# 课程分布配置
easy_ratio = 0.80
medium_ratio = 0.15
exploration_ratio = 0.05
skip_levels = [3, 4]

# 地形类型比例配置（对应8种类型）
terrain_proportions = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]

print("="*60)
print("地形分布测试")
print("="*60)

# === 测试1: 难度级别分配（行） ===
print("\n【测试1】难度级别分配（行）")
print("-"*60)

num_easy = int(num_envs * easy_ratio)
num_medium = int(num_envs * medium_ratio)
num_exploration = num_envs - num_easy - num_medium

terrain_levels = torch.zeros(num_envs, dtype=torch.long)

# 1. 80% 环境：难度 0-1
easy_levels = torch.randint(0, 2, (num_easy,))
terrain_levels[:num_easy] = easy_levels

# 2. 15% 环境：难度 2
terrain_levels[num_easy:num_easy+num_medium] = 2

# 3. 5% 环境：视觉探索（跳过3-4）
if num_exploration > 0:
    available_levels = [i for i in range(num_rows) if i not in skip_levels]
    exploration_levels = torch.tensor(
        [available_levels[i % len(available_levels)] 
         for i in torch.randint(0, len(available_levels), (num_exploration,))]
    )
    terrain_levels[num_easy+num_medium:] = exploration_levels

# 随机打乱
perm = torch.randperm(num_envs)
terrain_levels = terrain_levels[perm]

print(f"配置:")
print(f"  简单地形 (0-1): {num_easy} 个环境 ({easy_ratio:.1%})")
print(f"  中等地形 (2):   {num_medium} 个环境 ({medium_ratio:.1%})")
print(f"  视觉探索:       {num_exploration} 个环境 ({exploration_ratio:.1%})")
print(f"  🚫 跳过难度: {skip_levels}")

print(f"\n实际分布:")
for level in range(num_rows):
    count = (terrain_levels == level).sum().item()
    percentage = count / num_envs * 100
    if count > 0:
        print(f"  难度 {level}: {count:4d} 个环境 ({percentage:5.2f}%)")
    else:
        marker = "🚫" if level in skip_levels else ""
        print(f"  难度 {level}: {count:4d} 个环境 ({percentage:5.2f}%) {marker}")

# 检查跳过的难度
error_count = 0
for level in skip_levels:
    count = (terrain_levels == level).sum().item()
    if count > 0:
        print(f"❌ 错误: {count} 个环境出现在应该跳过的难度 {level}!")
        error_count += 1

if error_count == 0:
    print(f"✅ 通过: 所有跳过的难度都没有环境分配")


# === 测试2: 地形类型分配（列） ===
print("\n" + "="*60)
print("【测试2】地形类型分配（列）")
print("-"*60)

# 方法1: 原始均匀分配（错误）
print("\n方法1: 原始均匀分配 (错误方法)")
terrain_types_old = torch.div(
    torch.arange(num_envs), 
    (num_envs/num_cols), 
    rounding_mode='floor'
).to(torch.long)

print("分布:")
for col_idx in range(num_cols):
    count = (terrain_types_old == col_idx).sum().item()
    percentage = count / num_envs * 100
    print(f"  类型 {col_idx}: {count:4d} 个环境 ({percentage:5.2f}%)")

# 方法2: 按比例随机分配（正确）
print("\n方法2: 按比例随机分配 (正确方法)")
proportions = np.array(terrain_proportions)
proportions = proportions / proportions.sum()  # 归一化

terrain_types_new = torch.from_numpy(
    np.random.choice(num_cols, size=num_envs, p=proportions)
).to(torch.long)

print("配置比例:")
for col_idx in range(num_cols):
    print(f"  类型 {col_idx}: {proportions[col_idx]*100:5.2f}%")

print("\n实际分布:")
for col_idx in range(num_cols):
    count = (terrain_types_new == col_idx).sum().item()
    expected = proportions[col_idx] * 100
    actual = count / num_envs * 100
    diff = abs(actual - expected)
    marker = "✅" if diff < 3 else "⚠️"  # 3% 误差内接受
    print(f"  类型 {col_idx}: {count:4d} 个环境 (预期 {expected:5.2f}%, 实际 {actual:5.2f}%) {marker}")

print("\n" + "="*60)
print("测试完成")
print("="*60)
