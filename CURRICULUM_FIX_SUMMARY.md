# 课程学习修复总结

## 问题诊断

### 1. 地形难度增长过快 ⚠️
**症状**：机器人还没学会良好的速度跟随就被推到高难度地形，导致性能下降

**根本原因**（在 `legged_robot.py` 中）：
```python
# 原始晋级条件：走超过地形长度的 50% 就晋级
move_up = distance > self.terrain.env_length / 2  # 只需走 4m

# 原始降级条件：走不到目标距离的 50% 才降级
move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5)
```

**问题分析**：
- ✗ 晋级门槛太低：只需走 4m（地形长度 8m 的一半）
- ✗ 降级门槛太高：当速度跟随不好时，目标距离很小，容易触发降级
- ✗ 没有考虑速度跟随质量：即使跟随很差也会晋级
- ✗ 结果：机器人在低难度地形来回震荡，或被快速推到高难度

---

### 2. 命令课程增长过慢 ⚠️
**症状**：速度命令范围长期停留在初始值 `[0, 0.3]`，无法练习更高速度

**根本原因**（在 `legged_robot.py` 中）：
```python
# 原始晋级条件：需要 tracking reward 达到 80% 才扩展命令范围
if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
    self.command_ranges["lin_vel_x"][1] += 0.5  # 步长过大
```

**问题分析**：
- ✗ 阈值过高（80%）：很难达到，导致命令范围长期不变
- ✗ 步长过大（0.5 m/s）：一旦达标，速度突然增加过多
- ✗ 只扩展前进，不扩展后退：后退能力得不到训练
- ✗ 形成恶性循环：低速度 → 难以提升 → 命令不扩展 → 无法练习高速

---

### 3. 命令采样方向单一 ⚠️
**症状**：机器人看起来只向 X 正方向运动，缺少多方向训练

**根本原因**（在 `sirius_joystick.py` 中）：
```python
# 原始采样：90% 前进，10% 后退
forward_mask = direction_selector < 0.9

# 原始模式：80% 直行，20% 转向
straight_mode = mode_selector < 0.8
self.commands[env_ids[straight_mode], 2] *= 0.3  # 角速度大幅降低

# 横向速度范围过小
lin_vel_y = [-0.1, 0.1]  # 只有 ±0.1 m/s
```

**问题分析**：
- ⚠️ 前进偏向过强（90%）：缺少后退和横向运动训练
- ⚠️ 直行模式过多（80%）：角速度被大幅降低，转向能力不足
- ⚠️ 横向速度范围过小：即使采样了 Y 方向，幅度也很小
- ⚠️ 结果：机器人主要学会直线前进，缺少多方向泛化能力

---

## 修复方案

### ✅ 修复 1：调整地形课程更新逻辑

**文件**：`sirius_curriculum_config.py`

**新增方法**：`_update_terrain_curriculum()`

```python
def _update_terrain_curriculum(self, env_ids):
    """
    更严格的晋级条件 + 更宽松的降级条件
    """
    # 计算走过的距离
    distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
    
    # ✅ 晋级条件（更严格）：
    # 1. 走到地形长度的 70% 以上（原来 50%）
    # 2. 且 tracking reward > 40%（新增质量检查）
    move_up_distance = distance > (self.terrain.env_length * 0.7)
    tracking_quality = self.episode_sums["tracking_lin_vel"][env_ids] / episode_length
    move_up = move_up_distance & (tracking_quality > 0.4 * reward_scale)
    
    # ✅ 降级条件（更宽松）：
    # 走不到地形长度的 30% 才降级（原来需要走完目标距离的 50%）
    move_down = (distance < self.terrain.env_length * 0.3) & (~move_up)
    
    # 更新难度
    self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
```

**效果**：
- ✅ 晋级更困难：需要走得更远 + 速度跟随更好
- ✅ 降级更容易：给机器人更多时间在当前难度学习
- ✅ 防止震荡：增加了稳定性，减少频繁升降级
- ✅ 调试友好：每 1000 步打印平均难度

---

### ✅ 修复 2：降低命令课程阈值

**文件**：`sirius_curriculum_config.py`

**配置修改**：
```python
class commands(SiriusFlatCfg.commands):
    curriculum = True
    max_curriculum = 0.8
    max_reverse_curriculum = 0.1  # ✅ 新增：后退速度上限
    min_forward_speed = 0.2       # ✅ 降低到 0.2（原来 0.3）
    curriculum_step = 0.1         # ✅ 步长减小到 0.1（原来 0.15）
    curriculum_threshold = 0.5    # ✅ 新增：阈值降低到 50%（原来 80%）
```

**新增方法**：`update_command_curriculum()`

```python
def update_command_curriculum(self, env_ids):
    """
    更低的阈值 + 更小的步长 + 同时扩展前进和后退
    """
    avg_tracking_reward = torch.mean(self.episode_sums["tracking_lin_vel"]) / self.max_episode_length
    target_reward = 0.5 * self.reward_scales["tracking_lin_vel"]  # ✅ 50% 阈值
    
    if avg_tracking_reward > target_reward:
        step = 0.1  # ✅ 更小步长
        
        # ✅ 扩展前进速度
        self.command_ranges["lin_vel_x"][1] = np.clip(
            self.command_ranges["lin_vel_x"][1] + step,
            0., 0.8
        )
        
        # ✅ 扩展后退速度（新增）
        self.command_ranges["lin_vel_x"][0] = np.clip(
            self.command_ranges["lin_vel_x"][0] - step * 0.5,
            -0.1, 0.
        )
```

**效果**：
- ✅ 命令范围能更快扩展：50% 阈值比 80% 容易达到
- ✅ 扩展更平滑：0.1 m/s 步长避免突变
- ✅ 前后均衡：同时训练前进和后退能力
- ✅ 有上限保护：防止速度过快导致不稳定

---

### ✅ 修复 3：改进命令采样策略

**文件**：`sirius_joystick.py`

**方法修改**：`_resample_commands()`

```python
def _resample_commands(self, env_ids):
    """
    全方向随机采样 + 更平衡的运动模式
    """
    # ✅ 更平衡的前后采样：70% 前进，30% 后退（原来 90%/10%）
    forward_mask = direction_selector < 0.7
    
    # ✅ 横向速度全范围采样（配置已扩展到 ±0.3）
    self.commands[env_ids, 1] = torch_rand_float(
        self.command_ranges["lin_vel_y"][0],  # -0.3
        self.command_ranges["lin_vel_y"][1],  # +0.3
        (len(env_ids), 1)
    )
    
    # ✅ 更平衡的运动模式：
    # - 60% 直行为主：角速度降低到 30%
    # - 30% 混合模式：线速度和角速度都正常
    # - 10% 转向为主：线速度降低到 40%
    mode_selector = torch.rand(len(env_ids))
    straight_mode = mode_selector < 0.6
    mixed_mode = (mode_selector >= 0.6) & (mode_selector < 0.9)
    turning_mode = mode_selector >= 0.9
    
    # ✅ 过滤过小的命令（阈值降低到 0.15，原来 0.2）
    small_cmd_mask = lin_vel_norm < 0.15
    self.commands[env_ids[small_cmd_mask], :2] = 0.0
```

**配置修改**：
```python
class ranges:
    lin_vel_x = [-0.1, 0.3]
    lin_vel_y = [-0.3, 0.3]  # ✅ 从 ±0.1 扩展到 ±0.3
    ang_vel_yaw = [-0.6, 0.6]
```

**效果**：
- ✅ 支持全方向运动：X/Y 速度都能大范围采样
- ✅ 前后更平衡：30% 后退训练（原来 10%）
- ✅ 横向运动增强：±0.3 m/s 范围（原来 ±0.1）
- ✅ 混合模式增加：30% 同时训练线速度和角速度
- ✅ 转向训练保留：10% 专注转向控制

---

## 修改文件清单

### 1. `sirius_curriculum_config.py`
- ✅ 添加 `import numpy as np`
- ✅ 修改 `commands` 配置
  - `min_forward_speed`: 0.3 → 0.2
  - `curriculum_step`: 0.15 → 0.1
  - `curriculum_threshold`: 新增 0.5
  - `lin_vel_y`: [-0.1, 0.1] → [-0.3, 0.3]
- ✅ 新增 `_update_terrain_curriculum()` 方法
- ✅ 新增 `update_command_curriculum()` 方法

### 2. `sirius_joystick.py`
- ✅ 修改 `_resample_commands()` 方法
  - 前进比例：90% → 70%
  - 直行模式：80% → 60%
  - 新增混合模式：30%
  - 转向模式：20% → 10%
  - 小命令阈值：0.2 → 0.15

---

## 预期效果

### 地形课程
- 🎯 机器人会在低难度地形停留更久
- 🎯 只有速度跟随良好时才晋级到更高难度
- 🎯 减少在不同难度间频繁震荡
- 🎯 平均地形难度增长更平缓

### 命令课程
- 🎯 速度范围能更快扩展（50% 阈值）
- 🎯 扩展更平滑（0.1 m/s 步长）
- 🎯 同时训练前进和后退能力
- 🎯 预计在 500-1000 次迭代内达到最大速度

### 命令采样
- 🎯 支持全方向运动（X/Y/旋转）
- 🎯 后退和横向运动训练增加
- 🎯 运动模式更多样化
- 🎯 策略泛化能力更强

---

## 使用建议

### 训练监控
建议在 TensorBoard 中监控以下指标：

1. **地形课程**：
   - `Command/terrain_level`：平均地形难度
   - 期望：缓慢平稳上升，不要快速震荡

2. **命令课程**：
   - `Command/max_lin_vel`：最大速度命令
   - `Command/min_lin_vel`：最小速度命令（后退）
   - 期望：逐步扩展，达到 [±0.1, 0.8]

3. **速度跟随**：
   - `Train/mean_reward`：总奖励
   - `Rewards/tracking_lin_vel`：速度跟随奖励
   - 期望：随着课程进展稳步提升

4. **运动多样性**：
   - 检查命令分布是否覆盖各个方向
   - 可以添加自定义日志记录命令统计

### 调试输出
代码会自动每 1000 步打印课程进度：
```
[Terrain Curriculum] Avg level: 2.34, Max: 5/10
[Command Curriculum] Extended lin_vel_x range to [-0.05, 0.50]
  Avg tracking reward: 6.234 > 5.000
```

### 如果还是太难/太简单
可以微调以下参数：

**地形课程更慢**：
```python
# 在 _update_terrain_curriculum 中
move_up_distance = distance > (self.terrain.env_length * 0.8)  # 提高到 80%
tracking_threshold = 0.5  # 提高到 50%
```

**命令课程更快**：
```python
# 在配置中
curriculum_threshold = 0.4  # 降低到 40%
curriculum_step = 0.15      # 增大到 0.15
```

**增加运动多样性**：
```python
# 在配置中
lin_vel_y = [-0.4, 0.4]    # 扩展到 ±0.4
ang_vel_yaw = [-0.8, 0.8]  # 扩展到 ±0.8
```

---

## 下一步

1. **重新开始训练**（推荐）：
   ```bash
   cd legged_gym
   python scripts/train.py --task=sirius_curriculum --num_envs=512 --headless
   ```

2. **监控训练进度**：
   ```bash
   tensorboard --logdir=logs/sirius_curriculum
   ```

3. **观察改进**：
   - 前 500 次迭代：应该在低难度地形，命令范围逐步扩展
   - 500-1500 次迭代：地形难度逐步提升，速度跟随改善
   - 1500+ 次迭代：达到中高难度，命令范围接近最大值

4. **如有问题**：
   - 检查 TensorBoard 曲线是否平滑
   - 查看终端的调试输出
   - 根据上面的"调试"部分微调参数

---

**修改日期**：2025年11月14日
**修改人**：GitHub Copilot
