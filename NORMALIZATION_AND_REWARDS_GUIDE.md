# 📊 观测归一化与奖励调优指南

## 🎯 目的
理解观测如何归一化以及奖励的量级，以便正确调整奖励权重（scales）。

---

## 📐 观测归一化（Normalization）

### 观测空间缩放因子 (obs_scales)

```python
class normalization:
    class obs_scales:
        lin_vel = 2.0        # 线速度归一化：原始值 (m/s) × 2.0
        ang_vel = 0.25       # 角速度归一化：原始值 (rad/s) × 0.25
        dof_pos = 1.0        # 关节位置归一化：原始值 (rad) × 1.0
        dof_vel = 0.05       # 关节速度归一化：原始值 (rad/s) × 0.05
        height_measurements = 5.0  # 高度测量归一化（不再使用）
    clip_observations = 100.  # 观测裁剪范围 [-100, 100]
    clip_actions = 100.       # 动作裁剪范围 [-100, 100]
```

### 实际观测构成（45维本体）

```python
# sirius_joystick.py: compute_observations()
self.obs_buf = torch.cat((
    self.base_ang_vel  * self.obs_scales.ang_vel,           # [0:3]   = 角速度 × 0.25
    self.projected_gravity,                                  # [3:6]   = 投影重力 (无缩放)
    self.commands[:, :3] * self.commands_scale,              # [6:9]   = 命令 (见下方)
    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,  # [9:21]  = 关节位置偏差 × 1.0
    self.dof_vel * self.obs_scales.dof_vel,                 # [21:33] = 关节速度 × 0.05
    self.actions                                             # [33:45] = 上一步动作 (无缩放)
), dim=-1)
```

### 命令缩放 (commands_scale)

```python
# commands_scale = [lin_vel_scale, lin_vel_scale, ang_vel_scale]
self.commands_scale = [2.0, 2.0, 0.25]

# 因此命令在观测中：
commands[:, 0] × 2.0   # lin_vel_x 命令 (m/s) × 2.0
commands[:, 1] × 2.0   # lin_vel_y 命令 (m/s) × 2.0
commands[:, 2] × 0.25  # ang_vel_yaw 命令 (rad/s) × 0.25
```

---

## 🎁 奖励函数分析

### 1. 跟踪奖励（Tracking Rewards）

#### `_reward_tracking_lin_vel` - 线速度跟踪
```python
lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
```

**量级分析**：
- **误差**: `(cmd - actual)²`，单位 (m/s)²
- **典型值**: 
  - 完美跟踪（误差=0）: reward = 1.0
  - 误差=0.25 m/s, tracking_sigma=0.25: reward = exp(-1) ≈ 0.37
  - 误差=0.5 m/s: reward = exp(-4) ≈ 0.02
- **当前权重**: `tracking_lin_vel = 1.0`
- **建议**: 这是主要奖励，保持 1.0 作为基准

#### `_reward_tracking_ang_vel` - 角速度跟踪
```python
ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
```

**量级分析**：
- **误差**: `(cmd - actual)²`，单位 (rad/s)²
- **典型值**: 与线速度类似，范围 [0, 1]
- **当前权重**: `tracking_ang_vel = 0.5`
- **建议**: 0.5 表示角速度跟踪是线速度的一半重要性

---

### 2. 稳定性惩罚（Stability Penalties）

#### `_reward_orientation` - 姿态惩罚
```python
return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
```

**量级分析**：
- **值**: `gravity_x² + gravity_y²`
- **典型值**:
  - 完全直立: 0.0
  - 倾斜5°: ≈ 0.0076
  - 倾斜15°: ≈ 0.067
  - 倾斜30°: ≈ 0.25
- **当前权重**: `orientation = -1.5`
- **实际惩罚**: -1.5 × 0.067 ≈ **-0.1** (倾斜15°)
- **建议**: 
  - 保持 -1.5 用于一般地形
  - 如果机器人过于保守，降低到 -1.0 或 -0.75

#### `_reward_base_height` - 高度惩罚
```python
terrain_height = torch.mean(self.measured_heights, dim=1)
base_height_above_terrain = self.root_states[:, 2] - terrain_height
return torch.square(base_height_above_terrain - self.cfg.rewards.base_height_target)
```

**量级分析**：
- **值**: `(height_error)²`，单位 m²
- **base_height_target**: 0.445 m（站立高度）
- **典型值**:
  - 完美高度: 0.0
  - 偏差 5cm: 0.0025
  - 偏差 10cm: 0.01
  - 偏差 20cm: 0.04
- **当前权重**: `base_height = -2.5`
- **实际惩罚**: -2.5 × 0.01 ≈ **-0.025** (偏差10cm)
- **建议**:
  - **-2.5** 较强，适合要求严格保持高度
  - 如果在复杂地形上卡住，降低到 **-1.0** 或 **-0.5**

#### `_reward_lin_vel_z` - 垂直速度惩罚
```python
return torch.square(self.base_lin_vel[:, 2])
```

**量级分析**：
- **值**: `vz²`，单位 (m/s)²
- **典型值**:
  - 静止: 0.0
  - vz = 0.1 m/s: 0.01
  - vz = 0.3 m/s: 0.09
  - vz = 0.5 m/s: 0.25
- **当前权重**: `lin_vel_z = -0.075`
- **实际惩罚**: -0.075 × 0.09 ≈ **-0.007** (vz=0.3)
- **建议**: 适中，保持 -0.075

#### `_reward_ang_vel_xy` - 横滚/俯仰角速度惩罚
```python
return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
```

**量级分析**：
- **值**: `ωx² + ωy²`，单位 (rad/s)²
- **典型值**:
  - 稳定: 0.0
  - ω = 0.1 rad/s: 0.01
  - ω = 0.5 rad/s: 0.25
- **当前权重**: `ang_vel_xy = -0.05`
- **实际惩罚**: -0.05 × 0.01 ≈ **-0.0005** (ω=0.1)
- **建议**: 较弱，可以增加到 **-0.1** 如果需要更稳定

---

### 3. 平滑性惩罚（Smoothness Penalties）

#### `_reward_action_rate` - 动作变化率惩罚
```python
return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

**量级分析**：
- **值**: `Σ(Δaction)²`，对所有12个关节
- **典型值**:
  - 无变化: 0.0
  - 平均变化 0.1/joint: 0.12 (0.01×12)
  - 平均变化 0.3/joint: 1.08
- **当前权重**: `action_rate = -0.01`
- **实际惩罚**: -0.01 × 0.12 ≈ **-0.0012**
- **建议**: 
  - 保持 -0.01 用于平滑运动
  - 如果动作过于僵硬，降低到 **-0.005**

#### `_reward_dof_acc` - 关节加速度惩罚
```python
return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
```

**量级分析**：
- **值**: `Σ(acc)²`，单位 (rad/s²)²
- **dt**: 0.02s (控制频率 50Hz)
- **典型值**:
  - Δvel = 0.1 rad/s: acc = 5 rad/s², acc² = 25
  - 12个关节: ≈ 300
- **当前权重**: 通常 ≈ **-2.5e-7**（基类默认）
- **实际惩罚**: -2.5e-7 × 300 ≈ **-0.000075**
- **建议**: 非常弱，主要用于数值稳定性

---

### 4. 碰撞与限位惩罚（Collision & Limit Penalties）

#### `_reward_collision` - 碰撞惩罚
```python
return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
```

**量级分析**：
- **值**: 碰撞body的数量（离散值 0, 1, 2, ...）
- **典型值**: 0（无碰撞）或 1-2（有碰撞）
- **当前权重**: `collision = -1.0`
- **实际惩罚**: -1.0 × 1 = **-1.0** (单次碰撞)
- **建议**: 
  - **-1.0** 适中
  - 增加到 **-2.0** 如果机器人过于鲁莽
  - 降低到 **-0.5** 如果在复杂地形上过于保守

#### `_reward_stumble` - 绊倒惩罚
```python
return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >
     5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
```

**量级分析**：
- **值**: 布尔值 (0 或 1)
- **条件**: 水平力 > 5 × 垂直力（脚踢到东西）
- **当前权重**: `stumble = -2.5`
- **实际惩罚**: -2.5 × 1 = **-2.5** (绊倒时)
- **建议**: 
  - **-2.5** 强惩罚，适合崎岖地形
  - 保持此值以鼓励抬脚

---

### 5. 步态奖励（Gait Rewards）

#### `_reward_feet_air_time` - 腾空时间奖励
```python
first_contact = (self.feet_air_time > 0.) * contact_filt
rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1
return rew_airTime
```

**量级分析**：
- **值**: `Σ(air_time - 0.5)` 对首次接触的脚
- **典型值**:
  - 短步（air_time=0.2s）: -0.3
  - 正常步（air_time=0.5s）: 0.0
  - 长步（air_time=1.0s）: 0.5
  - 2脚着地: ≈ 1.0
- **当前权重**: `feet_air_time = 2.0`
- **实际奖励**: 2.0 × 0.5 = **+1.0** (长步)
- **建议**: 
  - **2.0** 较强，鼓励大步快走
  - 如果步态不自然，降低到 **1.0**

#### `_reward_stand_still` - 静止惩罚
```python
return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
```

**量级分析**：
- **值**: `Σ|joint_error|` 当命令接近0时
- **典型值**: 0（运动中）或 0.5-2.0（静止但有姿态）
- **当前权重**: `stand_still = -0.25`
- **实际惩罚**: -0.25 × 1.0 = **-0.25**
- **建议**: 保持 -0.25，防止不动

---

### 6. 其他奖励

#### `_reward_posture` - 姿态奖励
```python
# 通常是关节位置接近默认配置
return -torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
```

**量级分析**：
- **值**: `-Σ(pos_error)²`
- **典型值**: -0.1 到 -2.0
- **当前权重**: `posture = 1.0`
- **实际奖励**: 1.0 × (-0.5) = **-0.5**
- **建议**: 鼓励保持默认姿态

#### `_reward_termination` - 终止惩罚
```python
return self.reset_buf * ~self.time_out_buf
```

**量级分析**：
- **值**: 1 (倒地) 或 0 (正常)
- **当前权重**: `termination = -5.0`
- **实际惩罚**: **-5.0** (倒地时)
- **建议**: 
  - **-5.0** 强烈避免倒地
  - 这是离散事件，通常设置较大值

---

## 🎚️ 当前配置总结（sirius_curriculum_config.py）

```python
class scales:
    # 跟踪奖励（主要目标）
    tracking_lin_vel = 1.0    # 线速度跟踪 [0, 1] → 最重要！
    tracking_ang_vel = 0.5    # 角速度跟踪 [0, 1] → 次要
    
    # 稳定性惩罚
    orientation = -1.5        # 姿态惩罚 ~[-0.1, 0] → 较强
    base_height = -2.5        # 高度惩罚 ~[-0.05, 0] → 强！
    lin_vel_z = -0.075        # 垂直速度 ~[-0.02, 0] → 适中
    ang_vel_xy = -0.05        # 横滚俯仰 ~[-0.01, 0] → 弱
    
    # 平滑性惩罚
    action_rate = -0.01       # 动作变化 ~[-0.01, 0] → 弱
    
    # 碰撞惩罚
    collision = -1.0          # 碰撞 {-1, 0} → 适中
    stumble = -2.5            # 绊倒 {-2.5, 0} → 强
    
    # 步态奖励
    feet_air_time = 2.0       # 腾空时间 [0, 2] → 强！
    stand_still = -0.25       # 静止惩罚 ~[-0.5, 0] → 弱
    
    # 姿态与终止
    posture = 1.0             # 姿态奖励 ~[-1, 0] → 适中
    termination = -5.0        # 倒地惩罚 {-5, 0} → 非常强！
```

---

## 📈 典型Reward范围估算

### 成功episode的奖励组成

```
总Reward ≈ tracking + stability + smoothness + gait + termination

正常行走（跟踪良好）:
  tracking_lin_vel:   +1.0 × 0.8 = +0.8
  tracking_ang_vel:   +0.5 × 0.7 = +0.35
  orientation:        -1.5 × 0.02 = -0.03
  base_height:        -2.5 × 0.005 = -0.01
  feet_air_time:      +2.0 × 0.4 = +0.8
  其他小惩罚:                     -0.1
  ────────────────────────────────────
  Total per step:                 ≈ +1.8
  
失败episode（倒地）:
  termination:        -5.0 × 1 = -5.0
  ────────────────────────────────────
  Total:                          ≈ -5.0 (instant)
```

---

## 🔧 调优建议

### 场景1：机器人过于保守（不敢走）
**症状**: 速度慢，不敢上坡/下坡

**调整**:
```python
orientation = -0.75      # 从 -1.5 降低
base_height = -1.0       # 从 -2.5 降低
collision = -0.5         # 从 -1.0 降低
```

### 场景2：机器人步态不自然
**症状**: 动作僵硬，小步快跑

**调整**:
```python
action_rate = -0.005     # 从 -0.01 降低
feet_air_time = 1.0      # 从 2.0 降低
posture = 0.5            # 从 1.0 降低
```

### 场景3：机器人容易倒地
**症状**: 频繁摔倒，训练不稳定

**调整**:
```python
orientation = -2.0       # 从 -1.5 增加
base_height = -3.0       # 从 -2.5 增加
ang_vel_xy = -0.1        # 从 -0.05 增加
termination = -10.0      # 从 -5.0 增加
```

### 场景4：机器人速度跟踪不好
**症状**: 经常跑错方向或速度

**调整**:
```python
tracking_lin_vel = 1.5   # 从 1.0 增加
tracking_ang_vel = 0.8   # 从 0.5 增加
# 同时降低其他惩罚让其专注跟踪：
orientation = -1.0       # 从 -1.5 降低
base_height = -1.5       # 从 -2.5 降低
```

---

## 🎓 调优原则

### 1. 相对重要性原则
**主要奖励**应该是最大的：
- `tracking_lin_vel = 1.0` 作为基准
- 其他奖励相对于它调整

### 2. 量级匹配原则
考虑奖励的**自然量级**：
- `orientation`: 量级 ~0.01-0.1 → scale 用 -1 到 -2
- `base_height`: 量级 ~0.001-0.01 → scale 用 -1 到 -5
- `collision`: 量级 ~1 (离散) → scale 用 -0.5 到 -2

### 3. 平衡原则
```
总正奖励 ≈ 总负惩罚 (在正常行走时)
```

如果惩罚过强，机器人会过于保守；  
如果奖励过强，机器人会忽略约束。

### 4. 调试流程
1. **先看 TensorBoard**: `Rewards/*` 查看各奖励项的实际值
2. **识别主导项**: 找出绝对值最大的几项
3. **单独调整**: 一次只调一个权重
4. **观察行为**: 看机器人行为是否改善
5. **记录结果**: 保存配置和对应的表现

---

## 📊 TensorBoard监控

训练时监控这些指标：

```python
# 主要奖励
Rewards/tracking_lin_vel  # 应该 > 0.5
Rewards/tracking_ang_vel  # 应该 > 0.3

# 关键惩罚
Rewards/base_height       # 应该 > -0.1
Rewards/orientation       # 应该 > -0.2
Rewards/collision         # 应该接近 0

# 总奖励
Rewards/total_reward      # 应该稳定上升
Episode/rew_mean          # 平均每step >1.0 为好
```

---

## 💡 实用技巧

### 快速诊断
```bash
# 训练几百步后，查看奖励分布
tensorboard --logdir=logs/

# 找到占主导的奖励项
# 如果某项的绝对值明显大于others，可能需要调整
```

### A/B测试
```python
# 保存基准配置
base_config = {
    'orientation': -1.5,
    'base_height': -2.5,
    ...
}

# 测试变体
test_config_A = base_config.copy()
test_config_A['orientation'] = -1.0

# 各训练1000步，比较Episode/rew_mean
```

---

**✅ 这份指南应该能帮助您理解和调优奖励权重！**

**建议**: 从当前配置开始训练，观察几百个episode后，根据机器人的实际行为和TensorBoard数据进行微调。
