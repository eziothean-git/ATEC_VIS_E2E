# 朝向对齐功能 - 调试与验证指南

## 🔍 问题现象
"依然还是边走边转的" - 机器人朝向与速度方向不一致

## 📊 系统工作原理

### heading_command = True 模式下的工作流程：

1. **命令采样阶段** (`_resample_commands`)
   ```python
   # 设置速度命令
   commands[:, 0] = vx  # 前进速度
   commands[:, 1] = vy  # 横向速度
   commands[:, 3] = atan2(vy, vx) ± 5°  # 朝向目标（与速度方向对齐）
   ```

2. **物理步进后** (`post_physics_step`)
   ```python
   # 基类自动将朝向转换为角速度
   current_heading = atan2(forward_y, forward_x)
   heading_error = commands[:, 3] - current_heading
   commands[:, 2] = 0.5 * clip(heading_error, -1, 1)  # 自动计算角速度
   ```

3. **观测构建** (`compute_observations`)
   ```python
   # 策略看到的是前3个命令（包含自动计算的角速度）
   obs = [..., commands[:, :3], ...]  # [vx, vy, ω_yaw]
   ```

4. **奖励计算** (`_reward_tracking_ang_vel`)
   ```python
   # 比较命令角速度与实际角速度
   error = (commands[:, 2] - base_ang_vel[:, 2])^2
   reward = exp(-error / sigma)
   ```

## 🐛 调试输出说明

### 输出1：命令采样
```
[_resample_commands] Called at step 0
  Resampling 512 environments
  heading_command mode: True

======================================================================
[Heading Alignment Check] Step 0
======================================================================
  Env    0: vel=(+0.350, +0.150), vel_dir=+23.2°, heading=+25.1°, diff= 1.9°
  Env    1: vel=(+0.280, -0.100), vel_dir=-19.7°, heading=-17.3°, diff= 2.4°
  Env    2: vel=(+0.250, +0.200), vel_dir=+38.7°, heading=+40.2°, diff= 1.5°
  Env    3: vel=(+0.300, +0.050), vel_dir= +9.5°, heading=+11.8°, diff= 2.3°
  Env    4: vel=(+0.220, -0.150), vel_dir=-34.3°, heading=-32.1°, diff= 2.2°
======================================================================
```

**检查点：**
- ✅ `heading_command mode: True` → 确认使用朝向模式
- ✅ `diff` 都在 5度以内 → 朝向与速度方向对齐
- ❌ 如果 `diff` > 10度 → 对齐逻辑有问题

### 输出2：朝向到角速度转换
```
======================================================================
[Heading to AngVel Conversion] Step 1000
======================================================================
  Env    0: target_heading=+25.1°, curr_heading=+10.5°, error=+14.6°, ang_vel_cmd=+0.127, actual_ang_vel=+0.105
  Env    1: target_heading=-17.3°, curr_heading=-15.2°, error= -2.1°, ang_vel_cmd=-0.018, actual_ang_vel=-0.015
  Env    2: target_heading=+40.2°, curr_heading=+38.7°, error= +1.5°, ang_vel_cmd=+0.013, actual_ang_vel=+0.010
  Env    3: target_heading=+11.8°, curr_heading= +9.5°, error= +2.3°, ang_vel_cmd=+0.020, actual_ang_vel=+0.018
  Env    4: target_heading=-32.1°, curr_heading=-34.3°, error= +2.2°, ang_vel_cmd=+0.019, actual_ang_vel=+0.016
======================================================================
```

**检查点：**
- ✅ `error` 逐渐减小 → 机器人在朝向目标旋转
- ✅ `ang_vel_cmd` 与 `error` 符号一致 → 转向方向正确
- ✅ `actual_ang_vel` 接近 `ang_vel_cmd` → 跟踪效果好
- ❌ 如果 `error` 持续很大 → 可能是跟踪增益（0.5）太小或奖励权重太低

## 🧪 快速验证步骤

### 1. 启动小规模训练（便于观察）
```bash
cd /home/eziothean/ATEC_VIS_E2E
python legged_gym/scripts/train.py --task=sirius_curriculum --num_envs=64
```

### 2. 观察前1000步的输出
查找以下关键信息：
- `[_resample_commands] Called` → 确认方法被调用
- `heading_command mode: True` → 确认配置生效
- `[Heading Alignment Check]` → 检查 diff < 5度
- `[Heading to AngVel Conversion]` → 检查转换逻辑

### 3. 如果仍然"边走边转"

#### 情况A：diff > 10度（对齐失败）
**原因：** 命令采样逻辑有问题
**解决：** 检查 `_resample_commands` 是否被正确覆盖

#### 情况B：diff < 5度，但 error 持续很大（跟踪失败）
**原因：** 朝向跟踪增益太低或奖励权重不足
**解决：** 
```python
# 选项1：增加转换增益（基类中）
commands[:, 2] = 1.0 * wrap_to_pi(commands[:, 3] - heading)  # 从0.5改为1.0

# 选项2：增加角速度跟踪奖励权重
class rewards:
    class scales:
        tracking_ang_vel = 15  # 从10增加到15
```

#### 情况C：diff < 5度，error < 10度，但仍然"边走边转"
**原因：** 可能是视觉观察导致的误判，或者横向速度命令太大
**解决：**
```python
# 减小横向速度范围
class ranges:
    lin_vel_y = [-0.15, 0.15]  # 从[-0.3, 0.3]减小到[-0.15, 0.15]
```

## 📝 清理缓存并重新运行

如果修改没有生效，执行：
```bash
cd /home/eziothean/ATEC_VIS_E2E

# 清理 Python 缓存
find legged_gym -name "*.pyc" -delete
find legged_gym -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 重新启动
python legged_gym/scripts/train.py --task=sirius_curriculum --num_envs=64
```

## ✅ 成功标志

当功能正常工作时，你应该看到：
1. ✅ 初始时：`diff < 5度`（命令对齐）
2. ✅ 训练中：`error` 逐渐减小（朝向跟踪改善）
3. ✅ 视觉上：机器人朝向运动方向，不再螃蟹步

## 🔧 如果还是不行

请提供以下信息：
1. 训练日志中的 `[Heading Alignment Check]` 输出
2. 训练日志中的 `[Heading to AngVel Conversion]` 输出
3. 观察到的具体现象（是否横向移动？转圈？）
