# 🎯 命令速度课程学习分析

## ❌ 问题现象

**训练 100 次迭代后：**
- `tracking_lin_vel` 奖励稳定在 **0.75**
- `tracking_ang_vel` 奖励稳定在 **0.35**
- **最大速度命令没有增加** ⚠️

---

## 🔍 根本原因分析

### 当前配置 (sirius_curriculum_config.py)

```python
class commands:
    curriculum = True
    max_curriculum = 0.6  # 最大线速度 (m/s)
    
    class ranges:
        lin_vel_x = [-0.1, 0.3]     # 初始前进速度范围
        lin_vel_y = [-0.1, 0.1]     # 初始横向速度范围
        ang_vel_yaw = [-0.8, 0.8]   # 初始转向速度范围
```

### 课程更新逻辑 (sirius_joystick.py:633)

```python
def update_command_curriculum(self, env_ids):
    """ Implements a curriculum of increasing commands """
    
    # 🚨 关键条件：只有当奖励超过 80% 时才增加速度范围
    if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
        self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
        self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
```

### 奖励函数 (跟踪奖励)

```python
def _reward_tracking_lin_vel(self):
    lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
    # tracking_sigma = 0.25
```

---

## 📊 数学分析：为什么速度没有增加？

### 1. **阈值条件**

课程更新需要：
```
平均奖励 / max_episode_length > 0.8 × reward_scale
```

当前设置：
- `reward_scale["tracking_lin_vel"] = 1.0`
- `max_episode_length = 1000` (假设)

因此需要：
```
episode_sum / 1000 > 0.8 × 1.0
episode_sum > 800
平均每步奖励 > 0.8
```

### 2. **您的实际表现**

```
当前奖励: 0.75 (每步平均)
```

**结论**：`0.75 < 0.8` ❌ **未达到阈值！**

---

## 🧮 奖励值计算

### 跟踪奖励公式

```
reward = exp(-error² / 0.25)
```

### 不同误差对应的奖励值

| 误差 (m/s) | error² | -error²/0.25 | exp(...) | 奖励值 |
|-----------|--------|--------------|----------|--------|
| 0.0       | 0.00   | 0.00         | e^0      | **1.00** |
| 0.1       | 0.01   | -0.04        | e^-0.04  | **0.96** |
| 0.2       | 0.04   | -0.16        | e^-0.16  | **0.85** |
| 0.25      | 0.0625 | -0.25        | e^-0.25  | **0.78** |
| 0.3       | 0.09   | -0.36        | e^-0.36  | **0.70** |
| 0.4       | 0.16   | -0.64        | e^-0.64  | **0.53** |
| 0.5       | 0.25   | -1.00        | e^-1.0   | **0.37** |

### 您的情况分析

```
tracking_lin_vel = 0.75  →  误差 ≈ 0.25-0.3 m/s
tracking_ang_vel = 0.35  →  误差 ≈ 0.5-0.6 rad/s
```

**这说明**：
1. 机器人能够跟踪命令，但有 **25-30cm/s 的误差**
2. 角速度跟踪更差，误差较大
3. **性能不够好，未达到 0.8 阈值，所以课程不推进**

---

## 🎯 问题的三个可能原因

### 原因 1：阈值设置过高 (80%)

**当前**：需要达到 0.8 才能推进

**实际情况**：
- 复杂地形 + 视觉学习 + 课程难度 → 达到 0.8 很困难
- 特别是在多种地形混合的情况下

### 原因 2：初始速度范围过小

**当前配置**：
```python
lin_vel_x = [-0.1, 0.3]  # 范围只有 0.4 m/s
```

**问题**：
- 机器人从未体验过高速运动
- 没有建立高速运动的策略
- 即使增加范围，可能也无法适应

### 原因 3：地形难度与速度课程不匹配

**当前配置**：
- 地形课程：`max_init_terrain_level = 2`（难度 0-2）
- 速度课程：需要在这些地形上达到 0.8 奖励

**问题**：
- 即使是难度 1-2 的地形，高速运动也很危险
- 机器人为了稳定，倾向于慢速运动
- **奖励冲突**：速度跟踪 vs 稳定性

---

## 🔧 解决方案

### 方案 1：降低课程推进阈值（推荐）

```python
# sirius_joystick.py: update_command_curriculum()

# 修改前：
if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:

# 修改后：
if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.7 * self.reward_scales["tracking_lin_vel"]:
#                                                                                           ^^^ 从 0.8 降到 0.7
```

**优点**：
- ✅ 简单直接
- ✅ 您的 0.75 已经接近 0.7，可以很快推进
- ✅ 允许在"足够好"而非"完美"时推进

**调试输出建议**：
```python
def update_command_curriculum(self, env_ids):
    avg_reward = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
    threshold = 0.7 * self.reward_scales["tracking_lin_vel"]
    
    # 🔍 添加调试输出
    if len(env_ids) > 0 and self.common_step_counter % self.max_episode_length == 0:
        print(f"🎯 Command Curriculum Check:")
        print(f"   Average tracking reward: {avg_reward:.3f}")
        print(f"   Threshold: {threshold:.3f}")
        print(f"   Current lin_vel_x range: [{self.command_ranges['lin_vel_x'][0]:.2f}, {self.command_ranges['lin_vel_x'][1]:.2f}]")
    
    if avg_reward > threshold:
        old_range = self.command_ranges["lin_vel_x"].copy()
        self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
        self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
        print(f"   ✅ CURRICULUM UPDATED: {old_range} → {self.command_ranges['lin_vel_x']}")
```

---

### 方案 2：增加初始速度范围（不推荐单独使用）

```python
# sirius_curriculum_config.py

class ranges:
    lin_vel_x = [-0.2, 0.5]     # 从 [-0.1, 0.3] 增加到 [-0.2, 0.5]
    lin_vel_y = [-0.2, 0.2]     # 从 [-0.1, 0.1] 增加到 [-0.2, 0.2]
    ang_vel_yaw = [-1.0, 1.0]   # 从 [-0.8, 0.8] 增加到 [-1.0, 1.0]
```

**问题**：
- ❌ 初始范围过大可能导致训练不稳定
- ❌ 机器人可能无法在复杂地形上完成高速运动
- ❌ 可能导致更多倒地

---

### 方案 3：分离地形课程和速度课程（高级）

```python
# sirius_curriculum_config.py

class commands:
    curriculum = True
    max_curriculum = 0.8  # 增加最大速度
    
    # 🆕 新增：速度课程阈值随地形难度调整
    curriculum_threshold_base = 0.7    # 基础阈值
    curriculum_threshold_decay = 0.05  # 每提升一级地形难度，阈值降低 0.05
    
    class ranges:
        lin_vel_x = [-0.1, 0.3]  # 保持初始范围
```

```python
# sirius_joystick.py: update_command_curriculum()

def update_command_curriculum(self, env_ids):
    # 🆕 根据平均地形难度动态调整阈值
    avg_terrain_level = torch.mean(self.terrain_levels[env_ids].float())
    threshold = self.cfg.commands.curriculum_threshold_base - avg_terrain_level * self.cfg.commands.curriculum_threshold_decay
    threshold = max(threshold, 0.5)  # 最低不低于 0.5
    
    avg_reward = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
    
    if avg_reward > threshold * self.reward_scales["tracking_lin_vel"]:
        # 更新速度范围...
```

**优点**：
- ✅ 在简单地形上要求高性能
- ✅ 在复杂地形上放宽要求
- ✅ 鼓励探索速度-地形的平衡

**缺点**：
- ❌ 实现复杂
- ❌ 需要额外调试

---

### 方案 4：使用多阶段课程（最保守）

```python
# sirius_joystick.py

def update_command_curriculum(self, env_ids):
    avg_reward = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
    
    # 🆕 阶段性课程
    current_max_vel = self.command_ranges["lin_vel_x"][1]
    
    # 阶段 1: 0.3 → 0.5 (阈值 0.7)
    if current_max_vel < 0.5 and avg_reward > 0.7 * self.reward_scales["tracking_lin_vel"]:
        self.command_ranges["lin_vel_x"][1] = 0.5
        print("📈 Stage 1 → 2: max_vel = 0.5 m/s")
    
    # 阶段 2: 0.5 → 0.7 (阈值 0.75)
    elif current_max_vel < 0.7 and avg_reward > 0.75 * self.reward_scales["tracking_lin_vel"]:
        self.command_ranges["lin_vel_x"][1] = 0.7
        print("📈 Stage 2 → 3: max_vel = 0.7 m/s")
    
    # 阶段 3: 0.7 → 0.8 (阈值 0.8)
    elif current_max_vel < 0.8 and avg_reward > 0.8 * self.reward_scales["tracking_lin_vel"]:
        self.command_ranges["lin_vel_x"][1] = 0.8
        print("📈 Stage 3 → 4: max_vel = 0.8 m/s")
```

---

## 💡 推荐方案组合

### 🎯 快速解决（立即可用）

```python
# 1. 降低阈值到 0.7
if avg_reward > 0.7 * self.reward_scales["tracking_lin_vel"]:

# 2. 添加调试输出
print(f"🎯 Command Curriculum: reward={avg_reward:.3f}, threshold=0.7, range={self.command_ranges['lin_vel_x']}")
```

### 🚀 长期优化（训练稳定后）

```python
# 1. 逐步增加 max_curriculum 到 0.8
max_curriculum = 0.8

# 2. 观察机器人在高速下的表现
# 3. 根据需要调整其他奖励权重（如降低 orientation 惩罚）
```

---

## 📈 预期效果

### 使用方案 1（阈值 0.7）后：

```
迭代 0-100:    lin_vel range = [-0.1, 0.3], reward ≈ 0.75
迭代 100-200:  lin_vel range = [-0.6, 0.8], reward ≈ 0.70  ⬅️ 应该会推进！
迭代 200-300:  lin_vel range = [-0.6, 0.8], reward ≈ 0.75  ⬅️ 适应后提升
迭代 300+:     稳定在最大速度
```

### 监控指标

在 TensorBoard 中添加：
```python
self.writer.add_scalar('Command/max_lin_vel', self.command_ranges["lin_vel_x"][1], self.tot_timesteps)
self.writer.add_scalar('Command/avg_tracking_reward', avg_reward, self.tot_timesteps)
self.writer.add_scalar('Command/curriculum_threshold', threshold, self.tot_timesteps)
```

---

## ⚠️ 注意事项

### 1. 速度增加可能导致：
- 更多倒地事件（termination）
- 姿态不稳定（orientation penalty）
- 碰撞增加（collision penalty）

### 2. 监控这些指标：
- `Episode/cumulative_reward` - 总奖励是否下降？
- `Episode/termination_rate` - 倒地率是否上升？
- `Rewards/collision` - 碰撞是否增加？

### 3. 如果出现问题：
- 回退阈值到 0.75
- 降低 `orientation`/`base_height` 惩罚
- 增加 `feet_air_time` 奖励（鼓励大步）

---

## 🎓 总结

### 核心问题
**0.8 阈值过高** → 您的 0.75 性能已经很好，但课程卡住了

### 推荐操作
1. **立即修改**: 阈值改为 0.7
2. **添加日志**: 监控课程推进过程
3. **观察训练**: 看速度范围是否开始增加
4. **调整奖励**: 如果倒地增加，降低稳定性惩罚

### 预期结果
- 100-200 迭代内，速度范围应该开始增加
- 最终达到 `[-0.6, 0.8]` m/s 的速度范围
- 机器人学会在复杂地形上快速移动

---

**✅ 建议先尝试方案 1（降低阈值），观察几百次迭代后再决定是否需要其他调整！**
