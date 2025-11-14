# 命令朝向对齐修复

## 🎯 问题描述

在课程训练中，命令采样时出现**机器人朝向与速度方向不一致**的情况：
- 速度命令指向某个方向（如前方偏右）
- 朝向命令可能指向完全不同的方向（如后方）
- 导致机器人需要"螃蟹步"或"侧向移动"来完成任务
- 这种不自然的运动模式不利于训练和实际部署

## 📋 原始实现

**位置：** `sirius_joystick.py::_resample_commands()`

```python
def _resample_commands(self, env_ids):
    # 采样线速度 x, y
    self.commands[env_ids, 0] = torch_rand_float(...)  # vx
    self.commands[env_ids, 1] = torch_rand_float(...)  # vy
    
    # 采样朝向 - 完全随机！
    if self.cfg.commands.heading_command:
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_ranges["heading"][0],  # -π
            self.command_ranges["heading"][1],  # π
            ...
        ).squeeze(1)
```

**问题：**
- 朝向在 `[-π, π]` 范围内完全随机
- 与速度方向 `atan2(vy, vx)` 完全解耦
- 机器人可能面向任意方向，同时要求向另一个方向移动

---

## ✅ 修复方案

### 🎯 核心思路

**朝向应该与速度方向对齐，只允许小范围偏差（±5度）**

这样：
1. 机器人大多数时候朝向运动方向
2. 允许小偏差模拟真实场景（如斜向观察障碍物）
3. 避免不自然的"侧向移动"

---

### 📝 实现代码

**位置：** `sirius_curriculum_config.py::SiriusCurriculum._resample_commands()`

```python
def _resample_commands(self, env_ids):
    # ============ 1. 采样线速度 ============
    # 前进/后退 x 方向
    # 横向 y 方向
    
    # ============ 2. 朝向命令：与速度方向对齐（±5度）============
    if self.cfg.commands.heading_command:
        # 计算速度方向角度
        vel_x = self.commands[env_ids, 0]
        vel_y = self.commands[env_ids, 1]
        vel_direction = torch.atan2(vel_y, vel_x)
        
        # 在速度方向正负5度范围内随机采样
        heading_offset_range = 5.0 * (3.14159265359 / 180.0)  # 5度 → 弧度
        heading_offset = torch_rand_float(
            -heading_offset_range,
            heading_offset_range,
            (len(env_ids), 1),
            device=self.device
        ).squeeze(1)
        
        # 设置朝向：速度方向 + 小偏移
        self.commands[env_ids, 3] = vel_direction + heading_offset
        
        # 归一化到 [-π, π]
        self.commands[env_ids, 3] = torch.atan2(
            torch.sin(self.commands[env_ids, 3]),
            torch.cos(self.commands[env_ids, 3])
        )
        
        # 特殊处理：速度接近零时
        vel_norm = torch.norm(self.commands[env_ids, :2], dim=1)
        zero_vel_mask = vel_norm < 0.1
        if zero_vel_mask.any():
            # 速度很小时，朝向在 [-5°, 5°] 范围（接近正前方）
            self.commands[env_ids[zero_vel_mask], 3] = torch_rand_float(
                -heading_offset_range,
                heading_offset_range,
                (zero_vel_mask.sum(), 1),
                device=self.device
            ).squeeze(1)
```

---

## 🔍 关键细节

### 1. **速度方向计算**
```python
vel_direction = torch.atan2(vel_y, vel_x)
```
- `atan2(y, x)` 返回从原点到 (x, y) 的角度
- 范围：`[-π, π]`
- 0° = 正前方 (x轴正向)
- 90° = 正左方 (y轴正向)

### 2. **偏移范围**
```python
heading_offset_range = 5.0 * (π / 180.0) ≈ 0.0873 rad
```
- ±5度 = ±0.0873 弧度
- 允许小范围观察偏差
- 不会导致不自然的"侧行"

### 3. **角度归一化**
```python
heading = torch.atan2(torch.sin(heading), torch.cos(heading))
```
- 将任意角度归一化到 `[-π, π]`
- 避免数值误差累积
- 确保角度连续性

### 4. **零速度处理**
```python
if vel_norm < 0.1:
    heading = rand([-5°, 5°])  # 朝向正前方附近
```
- 速度接近零时，速度方向无意义
- 设置朝向为正前方（0°）附近
- 避免除零或随机朝向

---

## 📊 效果对比

### 修复前：
```
示例命令：
- 速度: vx=0.5, vy=0.3 → 方向 ≈ 31°（右前方）
- 朝向: heading = -120°（左后方）

机器人行为：
❌ 面向左后方，同时要向右前方移动
❌ 需要"螃蟹步"或大幅度侧向移动
❌ 不自然，不利于学习
```

### 修复后：
```
示例命令：
- 速度: vx=0.5, vy=0.3 → 方向 ≈ 31°（右前方）
- 朝向: heading = 31° ± 5° = [26°, 36°]

机器人行为：
✅ 面向右前方（略有偏差）
✅ 向右前方移动
✅ 自然的运动模式，易于学习
```

---

## 🎓 设计理由

### 为什么允许 ±5度 而不是完全对齐？

1. **真实场景模拟**
   - 实际运动中，机器人可能需要观察侧方障碍物
   - 或者在转弯时提前调整朝向
   - 小偏差增加泛化能力

2. **避免过拟合**
   - 完全对齐（0度偏差）可能导致策略过于依赖精确对齐
   - 5度偏差模拟传感器噪声和控制误差

3. **平滑过渡**
   - 速度方向变化时，朝向也会相应变化
   - 小偏差允许更平滑的过渡

### 为什么是 5度？

- **太小（<3度）**：几乎无偏差，可能过于严格
- **太大（>10度）**：可能出现明显的"侧向移动"
- **5度**：经验值，平衡自然性和多样性

---

## 🧪 验证方法

### 1. 可视化命令分布
在训练日志中添加统计：
```python
# 计算朝向与速度方向的偏差
vel_direction = torch.atan2(self.commands[:, 1], self.commands[:, 0])
heading_error = torch.abs(self.commands[:, 3] - vel_direction)
heading_error = torch.min(heading_error, 2*π - heading_error)  # 处理周期性

print(f"Heading-Velocity alignment error: {heading_error.mean():.3f} rad ({heading_error.mean()*180/π:.1f}°)")
```

### 2. 观察训练行为
```bash
# 启动可视化训练
python train.py --task=sirius_curriculum --num_envs=64
```

观察：
- ✅ 机器人是否大多数时候朝向运动方向
- ✅ 是否还有不自然的"螃蟹步"
- ✅ 转向时是否流畅

### 3. TensorBoard 监控
```bash
tensorboard --logdir=logs/sirius_curriculum
```

关注指标：
- `tracking_ang_vel`：朝向跟随准确度
- `tracking_lin_vel`：速度跟随准确度
- 两者应该同时提高

---

## 📂 修改文件

### 主要修改：
1. **`sirius_curriculum_config.py`**
   - 添加 `torch_rand_float` 导入
   - 覆盖 `_resample_commands()` 方法
   - 实现朝向-速度对齐逻辑

### 受影响任务：
- ✅ `sirius_curriculum` - 带视觉的课程学习
- ✅ `sirius_curriculum_il` - IL 阶段1
- ✅ `sirius_curriculum_finetune` - IL 阶段2

### 不受影响任务：
- ⏸️ `sirius_teacher_curriculum` - 使用父类的 `_resample_commands`
  - 如果需要，可以同样覆盖这个方法

---

## 🔄 后续改进（可选）

### 1. 可配置的偏差范围
```python
class commands(SiriusFlatCfg.commands):
    heading_alignment_tolerance = 5.0  # 度数
```

### 2. 课程化的偏差范围
```python
# 初期训练：小偏差（±3度），易于学习
# 后期训练：大偏差（±10度），增强泛化
tolerance = 3.0 + 7.0 * curriculum_progress
```

### 3. 运动模式自适应
```python
# 高速直行：严格对齐（±2度）
# 低速转向：允许大偏差（±10度）
if lin_vel > 0.5:
    tolerance = 2.0
else:
    tolerance = 10.0
```

---

## 🚀 总结

这次修复解决了一个**关键的训练质量问题**：

- ❌ **修复前**：机器人可能面向任意方向，同时要求向另一方向移动
- ✅ **修复后**：机器人朝向与运动方向基本一致（±5度偏差）

**预期效果：**
1. 更自然的运动模式
2. 更快的训练收敛
3. 更好的实际部署性能
4. 提高速度和朝向跟随的协同性

**建议：**
- 重新开始训练 `sirius_curriculum` 任务
- 观察 `tracking_ang_vel` 和 `tracking_lin_vel` 是否同时提高
- 可视化验证运动是否更加自然
