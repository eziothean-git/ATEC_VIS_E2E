# 后退速度课程修复

## 🐛 问题描述

从 TensorBoard logs 发现：`min_lin_vel` 与 `max_lin_vel` 对称增长，但设计要求后退速度应该限制在 **≤ 0.15 m/s**。

### 修复前的问题

```python
# sirius_joystick.py line 697-698（修复前）
self.command_ranges["lin_vel_x"][0] = np.clip(
    self.command_ranges["lin_vel_x"][0] - 0.5, 
    -self.cfg.commands.max_curriculum,  # ❌ 使用相同的最大值
    0.
)
self.command_ranges["lin_vel_x"][1] = np.clip(
    self.command_ranges["lin_vel_x"][1] + 0.5, 
    0., 
    self.cfg.commands.max_curriculum
)
```

**问题**：
- 配置：`max_curriculum = 0.6` m/s
- 初始范围：`[-0.1, 0.3]`
- 第1次推进：`[-0.6, 0.6]` ❌（后退速度达到 -0.6 m/s！）
- 前进和后退速度对称增长

**为什么这是问题？**
1. **安全性**：后退速度过快（-0.6 m/s）不安全，机器人无法看到身后
2. **实际应用**：真实场景中后退只用于微调位置，不需要高速后退
3. **训练稳定性**：高速后退更容易失控摔倒

---

## ✅ 修复方案

### 1. 添加独立的后退速度限制

**配置文件修改** (`sirius_curriculum_config.py`):

```python
class commands(SiriusFlatCfg.commands):
    curriculum = True
    max_curriculum = 0.6  # 最大前进速度（m/s）
    max_reverse_curriculum = 0.15  # 🔧 最大后退速度（m/s）- 新增
    
    class ranges:
        lin_vel_x = [-0.1, 0.3]     # 初始范围
        lin_vel_y = [-0.1, 0.1]
        ang_vel_yaw = [-0.6, 0.6]
```

### 2. 修改课程推进逻辑

**环境文件修改** (`sirius_joystick.py` line 697-701):

```python
# If the tracking reward is above 70% of the maximum, increase the range of commands
if avg_reward > threshold:
    old_range = [self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1]]
    # 🔧 修复：后退速度限制为 -0.15 m/s，前进速度可达 0.6 m/s（非对称）
    max_reverse = getattr(self.cfg.commands, 'max_reverse_curriculum', 0.15)
    self.command_ranges["lin_vel_x"][0] = np.clip(
        self.command_ranges["lin_vel_x"][0] - 0.5, 
        -max_reverse,  # ✅ 使用独立的后退限制
        0.
    )
    self.command_ranges["lin_vel_x"][1] = np.clip(
        self.command_ranges["lin_vel_x"][1] + 0.5, 
        0., 
        self.cfg.commands.max_curriculum
    )
```

### 3. 更新调试输出

**环境文件修改** (`sirius_joystick.py` line 685-693):

```python
# 🔍 调试输出：每个 episode 结束时打印课程状态
if len(env_ids) > 0 and self.common_step_counter % self.max_episode_length == 0:
    max_reverse = getattr(self.cfg.commands, 'max_reverse_curriculum', 0.15)
    print(f"🎯 Command Curriculum Check (step {self.common_step_counter}):")
    print(f"   Average tracking reward: {avg_reward:.4f}")
    print(f"   Threshold (0.7×scale):   {threshold:.4f}")
    print(f"   Current lin_vel_x range: [{self.command_ranges['lin_vel_x'][0]:.2f}, {self.command_ranges['lin_vel_x'][1]:.2f}] m/s")
    print(f"   Max forward speed:       {self.cfg.commands.max_curriculum:.2f} m/s")
    print(f"   Max reverse speed:       {max_reverse:.2f} m/s")  # ✅ 显示后退限制
```

---

## 📊 修复效果对比

### 课程推进过程

| 迭代 | 修复前范围 | 修复后范围 | 说明 |
|------|-----------|-----------|------|
| **初始** | `[-0.1, 0.3]` | `[-0.1, 0.3]` | 相同 |
| **第1次推进** | `[-0.6, 0.6]` ❌ | `[-0.15, 0.6]` ✅ | 后退限制在 -0.15 |
| **第2次推进** | `[-0.6, 0.6]` ❌ | `[-0.15, 0.6]` ✅ | 后退已达上限 |

### TensorBoard 指标变化

**修复前** ❌:
```
Command/min_lin_vel:  -0.10 → -0.60 → -0.60
Command/max_lin_vel:   0.30 →  0.60 →  0.60
```
- 对称增长
- 后退速度过快

**修复后** ✅:
```
Command/min_lin_vel:  -0.10 → -0.15 → -0.15
Command/max_lin_vel:   0.30 →  0.60 →  0.60
```
- 非对称增长
- 后退速度安全限制

---

## 🎯 设计理念

### 为什么后退速度要限制？

1. **安全性**：
   - 机器人没有后视传感器
   - 高速后退容易撞到障碍物
   - 平衡控制更困难

2. **功能性**：
   - 后退主要用于：
     - 微调位置（< 0.1 m/s）
     - 避障后退（< 0.15 m/s）
   - 不需要高速后退功能

3. **训练效率**：
   - 限制不常用的运动模式
   - 专注于前进和转向能力
   - 减少训练中的摔倒

### 速度范围设计原则

```
前进速度：[-0.15, 0.60] m/s
         ▲          ▲
         │          │
      安全后退    正常行走速度
      微调位置
```

- **后退**：-0.15 m/s（慢速安全）
- **站立**：0.0 m/s（静止）
- **慢走**：0.3 m/s（初始训练）
- **快走**：0.6 m/s（最终目标）

---

## 🔍 验证方法

### 1. 查看终端输出

训练时应该看到：

```bash
🎯 Command Curriculum Check (step 256000):
   Average tracking reward: 0.7500
   Threshold (0.7×scale):   0.7000
   Current lin_vel_x range: [-0.10, 0.30] m/s
   Max forward speed:       0.60 m/s
   Max reverse speed:       0.15 m/s  # ✅ 显示后退限制

   ✅ CURRICULUM ADVANCED: lin_vel_x [-0.10, 0.30] → [-0.15, 0.60]
   #                                  ▲      ▲         ▲      ▲
   #                                  │      │         │      │
   #                               后退不会超过 -0.15    前进可达 0.6
```

### 2. 查看 TensorBoard

打开 TensorBoard：
```bash
tensorboard --logdir=logs/
```

检查指标：
- **Command/min_lin_vel**: 应该在 `-0.10 → -0.15` 之间停止
- **Command/max_lin_vel**: 应该在 `0.30 → 0.60` 继续增长
- 非对称曲线 ✅

### 3. 测试代码

```python
# 模拟课程推进
initial_range = [-0.1, 0.3]
max_forward = 0.6
max_reverse = 0.15

# 第1次推进
new_min = np.clip(initial_range[0] - 0.5, -max_reverse, 0.)
new_max = np.clip(initial_range[1] + 0.5, 0., max_forward)

print(f"第1次推进: [{initial_range[0]:.2f}, {initial_range[1]:.2f}] → [{new_min:.2f}, {new_max:.2f}]")
# 输出: 第1次推进: [-0.10, 0.30] → [-0.15, 0.60] ✅

# 第2次推进（已达上限）
new_min2 = np.clip(new_min - 0.5, -max_reverse, 0.)
new_max2 = np.clip(new_max + 0.5, 0., max_forward)

print(f"第2次推进: [{new_min:.2f}, {new_max:.2f}] → [{new_min2:.2f}, {new_max2:.2f}]")
# 输出: 第2次推进: [-0.15, 0.60] → [-0.15, 0.60] ✅（不再变化）
```

---

## 📝 总结

### 关键修改

1. ✅ 添加 `max_reverse_curriculum = 0.15` 配置
2. ✅ 使用独立的后退速度限制
3. ✅ 更新调试输出显示两个限制

### 预期效果

- **安全性提升**：后退速度不超过 -0.15 m/s
- **训练稳定**：减少高速后退导致的摔倒
- **符合设计**：前进-后退非对称（0.6 vs 0.15）
- **TensorBoard 可视化**：清晰看到非对称增长

### 向后兼容

使用 `getattr()` 提供默认值：
```python
max_reverse = getattr(self.cfg.commands, 'max_reverse_curriculum', 0.15)
```
- 如果配置文件没有 `max_reverse_curriculum`，默认使用 0.15
- 不会破坏旧的配置文件

---

## 🚀 下一步

重新开始训练并观察：

```bash
cd /home/eziothean/ATEC_VIS_E2E/legged_gym
python legged_gym/scripts/train.py --task=sirius_curriculum
```

监控指标：
- [ ] `Command/min_lin_vel` 在 -0.15 停止 ✅
- [ ] `Command/max_lin_vel` 增长到 0.60 ✅
- [ ] 终端输出显示非对称推进 ✅
- [ ] 摔倒率没有显著增加（后退限制带来的稳定性）

---

**修复完成！后退速度现在正确限制在 ≤ 0.15 m/s。**
