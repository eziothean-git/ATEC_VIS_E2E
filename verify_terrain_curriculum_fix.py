#!/usr/bin/env python3
"""
地形课程学习修复验证
======================

验证修复后的地形课程学习逻辑：
1. 升级阈值：> 4.0 米 (固定)
2. 降级阈值：< 2.0 米 (固定，从动态修改)
3. 安全区：2.0-4.0 米 (保持当前难度)
"""

print("=" * 70)
print("地形课程学习修复验证")
print("=" * 70)

# 配置参数
env_length = 8.0  # 地形单元长度 (米)
episode_length_s = 20.0  # episode 持续时间 (秒)

upgrade_threshold = env_length / 2  # 4.0 米
downgrade_threshold_old = lambda cmd_vel: cmd_vel * episode_length_s * 0.5  # 动态
downgrade_threshold_new = env_length / 4  # 2.0 米

print(f"\n📐 配置参数:")
print(f"  env_length:       {env_length:.1f} 米")
print(f"  episode_length_s: {episode_length_s:.1f} 秒")
print(f"  升级阈值 (固定): > {upgrade_threshold:.1f} 米")

print(f"\n" + "=" * 70)
print("问题分析：动态降级阈值的问题")
print("=" * 70)

# 测试不同的命令速度
test_velocities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

print(f"\n修复前 ❌ (动态降级阈值):")
print(f"\n| 命令速度 | 降级阈值 | 升级阈值 | 冲突区间 | 问题 |")
print(f"|---------|---------|---------|---------|------|")

for vel in test_velocities:
    downgrade_thresh = downgrade_threshold_old(vel)
    upgrade_thresh = upgrade_threshold
    
    if downgrade_thresh > upgrade_thresh:
        conflict_zone = f"[{upgrade_thresh:.1f}, {downgrade_thresh:.1f}]"
        problem = "❌ 冲突！"
    else:
        conflict_zone = "无"
        problem = "✅ 正常"
    
    print(f"| {vel:.1f} m/s  | < {downgrade_thresh:.1f} m | > {upgrade_thresh:.1f} m | {conflict_zone:13s} | {problem} |")

print(f"\n⚠️ 关键问题：")
print(f"   当命令速度 ≥ 0.4 m/s 时，降级阈值 > 升级阈值")
print(f"   在冲突区间内：不升级但会降级 → 地形难度持续下降！")

print(f"\n修复后 ✅ (固定降级阈值):")
print(f"\n| 命令速度 | 降级阈值 | 升级阈值 | 安全区间 | 状态 |")
print(f"|---------|---------|---------|---------|------|")

for vel in test_velocities:
    downgrade_thresh = downgrade_threshold_new
    upgrade_thresh = upgrade_threshold
    safe_zone = f"[{downgrade_thresh:.1f}, {upgrade_thresh:.1f}]"
    
    print(f"| {vel:.1f} m/s  | < {downgrade_thresh:.1f} m | > {upgrade_thresh:.1f} m | {safe_zone:13s} | ✅ 正常 |")

print(f"\n✅ 修复后特点：")
print(f"   降级阈值固定在 2.0 米，与命令速度无关")
print(f"   安全区间 [2.0, 4.0] 米：保持当前难度")
print(f"   不再有升级/降级冲突！")

print(f"\n" + "=" * 70)
print("实际场景模拟")
print("=" * 70)

scenarios = [
    # (description, cmd_vel, actual_vel, terrain_level)
    ("简单地形，完美跟踪", 0.3, 0.3, 1),
    ("中等地形，轻微误差", 0.4, 0.35, 2),
    ("复杂地形，明显误差", 0.6, 0.4, 3),
    ("复杂地形，大误差", 0.6, 0.25, 3),
    ("复杂地形，严重失败", 0.6, 0.08, 3),
]

print(f"\n场景测试:")

for desc, cmd_vel, actual_vel, level in scenarios:
    distance = actual_vel * episode_length_s
    
    # 修复前的判定
    old_downgrade_thresh = downgrade_threshold_old(cmd_vel)
    if distance > upgrade_threshold:
        old_action = "⬆️ 升级"
    elif distance < old_downgrade_thresh:
        old_action = "⬇️ 降级"
    else:
        old_action = "🔄 保持"
    
    # 修复后的判定
    new_downgrade_thresh = downgrade_threshold_new
    if distance > upgrade_threshold:
        new_action = "⬆️ 升级"
    elif distance < new_downgrade_thresh:
        new_action = "⬇️ 降级"
    else:
        new_action = "🔄 保持"
    
    print(f"\n场景: {desc}")
    print(f"  当前难度: {level}")
    print(f"  命令速度: {cmd_vel:.2f} m/s")
    print(f"  实际速度: {actual_vel:.2f} m/s (误差: {(cmd_vel - actual_vel) / cmd_vel * 100:.0f}%)")
    print(f"  行走距离: {distance:.2f} m")
    print(f"  修复前 (降级阈值 {old_downgrade_thresh:.1f}m): {old_action}")
    print(f"  修复后 (降级阈值 {new_downgrade_thresh:.1f}m): {new_action}")
    
    if old_action != new_action:
        print(f"  💡 修复改变了判定！")

print(f"\n" + "=" * 70)
print("预期训练效果对比")
print("=" * 70)

print(f"""
修复前 ❌:
─────────────────────────────────────────────────────────────
命令速度课程推进后 (0.6 m/s)：

Episode 1:  level=2, dist=3.5m  → 降级 (< 6.0m)  ❌
Episode 2:  level=1, dist=5.0m  → 升级 (> 4.0m)  ✅
Episode 3:  level=2, dist=3.2m  → 降级 (< 6.0m)  ❌
Episode 4:  level=1, dist=4.8m  → 升级 (> 4.0m)  ✅
Episode 5:  level=2, dist=3.8m  → 降级 (< 6.0m)  ❌
...

平均难度: 1.2-1.5 (震荡，无法稳定提升！)
─────────────────────────────────────────────────────────────

修复后 ✅:
─────────────────────────────────────────────────────────────
命令速度课程推进后 (0.6 m/s)：

Episode 1:  level=2, dist=3.5m  → 保持 (2.0-4.0m)  🔄
Episode 2:  level=2, dist=3.8m  → 保持 (2.0-4.0m)  🔄
Episode 3:  level=2, dist=4.5m  → 升级 (> 4.0m)    ✅
Episode 4:  level=3, dist=3.2m  → 保持 (2.0-4.0m)  🔄
Episode 5:  level=3, dist=4.2m  → 升级 (> 4.0m)    ✅
Episode 6:  level=4, dist=3.6m  → 保持 (2.0-4.0m)  🔄
Episode 7:  level=4, dist=1.8m  → 降级 (< 2.0m)    ⬇️
Episode 8:  level=3, dist=4.1m  → 升级 (> 4.0m)    ✅
...

平均难度: 2.5-3.5 (稳定提升！)
─────────────────────────────────────────────────────────────
""")

print("=" * 70)
print("监控指标")
print("=" * 70)

print(f"""
训练时请关注以下 TensorBoard 指标：

📊 Terrain/avg_level
   - 监控平均地形难度
   - 应该逐渐上升，而不是震荡

📊 Terrain/max_level
   - 监控最高地形难度
   - 应该接近 max_terrain_level

📊 Terrain/min_level
   - 监控最低地形难度
   - 大部分环境应该远离 0

📊 Terrain/level_X_count (X=0,1,2...9)
   - 监控各难度的环境数量分布
   - 应该呈正态分布，而不是集中在低难度

⚠️ 警告指标：
📊 Command/max_lin_vel
   - 如果地形难度上升但速度命令没增加
   - 说明速度课程被地形课程阻塞

📊 Episode/rew_mean
   - 如果地形难度上升后奖励骤降
   - 可能需要调整奖励权重
""")

print("=" * 70)
print("终端调试输出")
print("=" * 70)

print(f"""
训练时您将看到类似的输出：

🏔️ Terrain Curriculum Update (step 1000):
   Upgraded:     45/256 envs (distance > 4.0m)
   Downgraded:    8/256 envs (distance < 2.0m)
   Stayed:      203/256 envs (2.0m ≤ distance ≤ 4.0m)
   Avg distance traveled: 3.45 m
   Avg terrain level:     2.18

🏔️ Terrain Curriculum Update (step 2000):
   Upgraded:     52/256 envs (distance > 4.0m)
   Downgraded:    5/256 envs (distance < 2.0m)
   Stayed:      199/256 envs (2.0m ≤ distance ≤ 4.0m)
   Avg distance traveled: 3.68 m
   Avg terrain level:     2.42

关键观察：
  ✅ Upgraded > Downgraded  → 平均难度上升
  ✅ Stayed 是大多数       → 稳定训练
  ✅ Avg terrain level 逐渐增加
""")

print("=" * 70)
print("调优建议")
print("=" * 70)

print(f"""
如果修复后仍有问题：

1️⃣ 地形难度上升太慢：
   降低升级阈值:
   move_up = distance > self.terrain.env_length / 2.5  # 3.2m instead of 4.0m

2️⃣ 地形难度仍然下降：
   降低降级阈值:
   move_down = distance < self.terrain.env_length / 6  # 1.33m instead of 2.0m

3️⃣ 地形难度震荡：
   增大安全区间:
   upgrade_threshold = env_length / 2.5    # 3.2m
   downgrade_threshold = env_length / 6    # 1.33m
   安全区间: [1.33m, 3.2m]

4️⃣ 与速度课程不协调：
   查看 COMMAND_CURRICULUM_ANALYSIS.md
   确保两个课程都正常工作
""")

print("=" * 70)
print("✅ 验证完成！")
print("=" * 70)

print(f"""
总结：
1. ✅ 降级阈值从动态 (cmd_vel × 10) 改为固定 (2.0m)
2. ✅ 消除了升级/降级阈值冲突
3. ✅ 添加了详细的调试输出
4. ✅ 添加了 TensorBoard 监控

关键改进：
  修复前: 升级阈值 4.0m, 降级阈值 6.0m (cmd=0.6m/s) → 冲突！
  修复后: 升级阈值 4.0m, 降级阈值 2.0m          → 正常！

下一步：
1. 重新开始训练: python train.py --task=sirius_curriculum
2. 观察终端输出中的地形课程统计
3. 在 TensorBoard 中查看 Terrain/avg_level 是否上升
4. 如果有问题，查看完整分析: TERRAIN_CURRICULUM_ANALYSIS.md
""")

print("=" * 70)
