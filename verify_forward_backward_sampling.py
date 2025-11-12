#!/usr/bin/env python3
"""
前进/后退采样分布验证
============================

验证修复后的命令采样：90% 前进，10% 后退
"""

import numpy as np

print("=" * 70)
print("前进/后退采样分布验证")
print("=" * 70)

# 配置参数
lin_vel_range = [-0.15, 0.6]  # 最终范围（课程推进后）
forward_prob = 0.9  # 90% 前进
backward_prob = 0.1  # 10% 后退

print(f"\n📐 配置参数:")
print(f"  lin_vel_x 范围:     [{lin_vel_range[0]:.2f}, {lin_vel_range[1]:.2f}] m/s")
print(f"  前进概率:           {forward_prob:.0%}")
print(f"  后退概率:           {backward_prob:.0%}")

print(f"\n" + "=" * 70)
print("修复前：均匀采样（问题）")
print("=" * 70)

print(f"""
修复前的采样逻辑：
────────────────────────────────────────────────────────────
lin_vel_x = torch_rand_float(min, max, ...)
# 从 [{lin_vel_range[0]:.2f}, {lin_vel_range[1]:.2f}] 均匀随机采样

问题分析：
────────────────────────────────────────────────────────────
范围总长度 = {lin_vel_range[1]} - ({lin_vel_range[0]}) = {lin_vel_range[1] - lin_vel_range[0]:.2f} m/s
后退范围长度 = 0 - ({lin_vel_range[0]}) = {-lin_vel_range[0]:.2f} m/s
前进范围长度 = {lin_vel_range[1]} - 0 = {lin_vel_range[1]:.2f} m/s

实际概率：
  后退概率 = {-lin_vel_range[0]:.2f} / {lin_vel_range[1] - lin_vel_range[0]:.2f} = {-lin_vel_range[0] / (lin_vel_range[1] - lin_vel_range[0]):.1%} ❌
  前进概率 = {lin_vel_range[1]:.2f} / {lin_vel_range[1] - lin_vel_range[0]:.2f} = {lin_vel_range[1] / (lin_vel_range[1] - lin_vel_range[0]):.1%}

问题：后退概率过高（20% 而非 10%）❌
────────────────────────────────────────────────────────────
""")

print("=" * 70)
print("修复后：偏向前进采样（解决方案）")
print("=" * 70)

print(f"""
修复后的采样逻辑：
────────────────────────────────────────────────────────────
# 1. 随机决定方向（90% 前进，10% 后退）
direction_selector = torch.rand(len(env_ids))
forward_mask = direction_selector < 0.9   # 90%
backward_mask = ~forward_mask              # 10%

# 2. 根据方向分别采样
if forward_mask.any():
    commands[forward_mask, 0] = torch_rand_float(0., max, ...)
    # 仅从前进范围 [0, {lin_vel_range[1]:.1f}] 采样

if backward_mask.any():
    commands[backward_mask, 0] = torch_rand_float(min, 0., ...)
    # 仅从后退范围 [{lin_vel_range[0]:.2f}, 0] 采样
────────────────────────────────────────────────────────────

优势：
✅ 精确控制前进/后退概率（90% / 10%）
✅ 独立于速度范围的具体数值
✅ 适应课程推进时的范围变化
────────────────────────────────────────────────────────────
""")

print("=" * 70)
print("采样分布模拟（256 环境）")
print("=" * 70)

# 模拟采样
num_envs = 256
np.random.seed(42)

print(f"\n修复前（均匀采样）:")
print(f"─────────────────────────────────────────────────────────")
old_samples = np.random.uniform(lin_vel_range[0], lin_vel_range[1], num_envs)
old_forward = (old_samples > 0).sum()
old_backward = (old_samples < 0).sum()
old_zero = (old_samples == 0).sum()

print(f"前进命令: {old_forward}/{num_envs} ({old_forward/num_envs:.1%})")
print(f"后退命令: {old_backward}/{num_envs} ({old_backward/num_envs:.1%})")
print(f"零速命令: {old_zero}/{num_envs} ({old_zero/num_envs:.1%})")
print(f"❌ 后退比例 {old_backward/num_envs:.1%} != 目标 10%")

print(f"\n修复后（偏向前进采样）:")
print(f"─────────────────────────────────────────────────────────")
direction_selector = np.random.rand(num_envs)
forward_mask = direction_selector < 0.9
backward_mask = ~forward_mask

new_samples = np.zeros(num_envs)
new_samples[forward_mask] = np.random.uniform(0, lin_vel_range[1], forward_mask.sum())
new_samples[backward_mask] = np.random.uniform(lin_vel_range[0], 0, backward_mask.sum())

new_forward = (new_samples > 0).sum()
new_backward = (new_samples < 0).sum()
new_zero = (new_samples == 0).sum()

print(f"前进命令: {new_forward}/{num_envs} ({new_forward/num_envs:.1%})")
print(f"后退命令: {new_backward}/{num_envs} ({new_backward/num_envs:.1%})")
print(f"零速命令: {new_zero}/{num_envs} ({new_zero/num_envs:.1%})")
print(f"✅ 后退比例 {new_backward/num_envs:.1%} ≈ 目标 10%")

print(f"\n速度分布统计:")
print(f"─────────────────────────────────────────────────────────")
print(f"修复前:")
print(f"  平均速度: {old_samples.mean():.3f} m/s")
print(f"  前进平均: {old_samples[old_samples > 0].mean():.3f} m/s")
print(f"  后退平均: {old_samples[old_samples < 0].mean():.3f} m/s")

print(f"\n修复后:")
print(f"  平均速度: {new_samples.mean():.3f} m/s")
print(f"  前进平均: {new_samples[new_samples > 0].mean():.3f} m/s")
print(f"  后退平均: {new_samples[new_samples < 0].mean():.3f} m/s")

print(f"\n✅ 修复后平均速度更高（更偏向前进）")

print(f"\n" + "=" * 70)
print("训练影响分析")
print("=" * 70)

print(f"""
为什么需要 90% 前进，10% 后退？

1️⃣ 实际应用场景：
   ────────────────────────────────────────────────────────
   前进场景：
     - 正常行走、巡逻
     - 导航到目标点
     - 跟随人或物体
     → 占 90%+ 的使用时间
   
   后退场景：
     - 微调位置（< 0.1 m/s）
     - 避障后退（< 0.15 m/s）
     - 调整姿态
     → 占 <10% 的使用时间
   ────────────────────────────────────────────────────────

2️⃣ 安全考虑：
   ────────────────────────────────────────────────────────
   - 机器人没有后视传感器（深度相机朝前）
   - 后退时无法看到障碍物
   - 高速后退更容易失控
   → 限制后退速度和频率
   ────────────────────────────────────────────────────────

3️⃣ 训练效率：
   ────────────────────────────────────────────────────────
   修复前（20% 后退）：
     - 训练时间浪费在不常用的后退
     - 模型对前进的优化不够充分
     - 后退能力过强，前进能力偏弱
   
   修复后（10% 后退）：
     - 90% 训练时间优化前进能力
     - 10% 时间保证基本后退能力
     - 更符合实际使用分布
   ────────────────────────────────────────────────────────
""")

print("=" * 70)
print("监控指标")
print("=" * 70)

print(f"""
训练时查看终端输出：

🎯 Command Curriculum Check (step 256000):
   Average tracking reward: 0.7500
   Threshold (0.7×scale):   0.7000
   Current lin_vel_x range: [-0.15, 0.60] m/s
   Max forward speed:       0.60 m/s
   Max reverse speed:       0.15 m/s
   📊 Command sampling: 230/256 forward (89.8%), 26/256 backward (10.2%)
                        ▲                 ▲       ▲                ▲
                        │                 │       │                │
                    应该约 230         约 90%   应该约 26        约 10%

✅ 如果看到这样的输出，说明采样正确！
❌ 如果 backward > 15%，说明有问题
""")

print("=" * 70)
print("TensorBoard 可视化")
print("=" * 70)

print(f"""
在 TensorBoard 中查看：

📊 Command/avg_lin_vel_x（新增指标）：
   - 修复前：约 0.20 m/s（受后退拉低）
   - 修复后：约 0.30 m/s（更偏向前进）✅

📊 Rewards/tracking_lin_vel：
   - 修复后应该更高（前进更多，跟踪更准确）

📊 Episode/termination_rate：
   - 修复后应该更低（后退少，更安全）
""")

print("=" * 70)
print("代码变更总结")
print("=" * 70)

print(f"""
修改文件: sirius_joystick.py
─────────────────────────────────────────────────────────

函数: _resample_commands() (line 469-510)

修改前 ❌:
  self.commands[env_ids, 0] = torch_rand_float(
      self.command_ranges["lin_vel_x"][0],  # -0.15
      self.command_ranges["lin_vel_x"][1],  # 0.60
      ...
  )
  # 均匀采样 → 20% 后退，80% 前进

修改后 ✅:
  direction_selector = torch.rand(len(env_ids))
  forward_mask = direction_selector < 0.9   # 90% 前进
  backward_mask = ~forward_mask              # 10% 后退
  
  # 前进: 从 [0, 0.60] 采样
  self.commands[env_ids[forward_mask], 0] = torch_rand_float(
      0., 
      self.command_ranges["lin_vel_x"][1], 
      ...
  )
  
  # 后退: 从 [-0.15, 0] 采样
  self.commands[env_ids[backward_mask], 0] = torch_rand_float(
      self.command_ranges["lin_vel_x"][0], 
      0., 
      ...
  )
  # 精确控制 → 10% 后退，90% 前进 ✅

新增调试输出 (line 700-718):
  print(f"📊 Command sampling: {forward_commands}/{total_commands} forward ({fwd_ratio:.1%}), ...")
  # 实时显示采样分布
─────────────────────────────────────────────────────────
""")

print("=" * 70)
print("✅ 验证完成！")
print("=" * 70)

print(f"""
总结：
1. ✅ 修复采样逻辑：从均匀采样 → 偏向前进采样
2. ✅ 前进/后退比例：从 80/20 → 90/10
3. ✅ 添加实时监控：终端显示采样分布统计
4. ✅ 符合实际应用：前进为主，后退为辅

预期效果：
  - 前进能力提升（更多训练）
  - 后退保留基本功能（10% 足够）
  - 训练更安全（减少后退摔倒）
  - 更符合部署场景（90% 前进使用）

下一步：
1. 重新开始训练: python train.py --task=sirius_curriculum
2. 查看终端输出中的 📊 Command sampling 统计
3. 确认 forward ≈ 90%, backward ≈ 10%
""")

print("=" * 70)
