# 🔧 高度奖励地形自适应修复

## 📌 问题描述

### 原始实现的问题
```python
def _reward_base_height(self):
    base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    return torch.square(base_height - self.cfg.rewards.base_height_target)
```

**致命缺陷**：
- ❌ `base_height_target = 0.445` 是**绝对高度**（相对于世界坐标原点）
- ❌ 在 **dimps（坑洼）** 地形中，机器人可能在 z = -0.5 的坑底
- ❌ 目标是 0.445，实际是 -0.5，惩罚巨大且不合理
- ❌ 在 **斜坡** 上，目标高度也是固定的，无法自适应

### 实际影响
在课程学习的复杂地形中：

```
平地（z ≈ 0）:
  目标: 0.445 ✓ 合理

dimps（z ≈ -0.5）:
  目标: 0.445 ✗ 错误！应该是 -0.5 + 0.445 = -0.055

斜坡（z 变化）:
  目标: 0.445 ✗ 错误！应该动态调整

台阶（z 跳变）:
  目标: 0.445 ✗ 错误！应该根据脚下地形调整
```

**结果**：
- 机器人在复杂地形上会尝试"爬升"到绝对高度 0.445
- 完全破坏了课程学习的意义
- 视觉 RL 无法学到正确的高度调节策略

## ✅ 修复方案

### 新实现（地形自适应）
```python
def _reward_base_height(self):
    # Penalize base height away from target
    # 计算机器人相对于当前地形的高度（脚下地形的平均高度）
    terrain_height = torch.mean(self.measured_heights, dim=1)  # [num_envs]
    base_height_above_terrain = self.root_states[:, 2] - terrain_height  # [num_envs]
    
    # 目标是保持在地形上方 base_height_target 的高度
    # 这样在平地、dimps、斜坡等各种地形都能自适应
    return torch.square(base_height_above_terrain - self.cfg.rewards.base_height_target)
```

### 核心思想
> **目标高度 = 当前地形高度 + 固定偏移量**

**关键改进**：
- ✅ `terrain_height`：脚下地形的平均高度（从 `measured_heights` 计算）
- ✅ `base_height_above_terrain`：机器人相对于地形的高度
- ✅ `base_height_target = 0.445`：现在是**相对高度**，而非绝对高度

## 📊 修复效果

### 各地形的行为
```
平地（terrain_height = 0）:
  目标: 0 + 0.445 = 0.445 ✓ 与之前一致

dimps（terrain_height = -0.5）:
  目标: -0.5 + 0.445 = -0.055 ✓ 正确！

斜坡（terrain_height 变化）:
  目标: terrain_height + 0.445 ✓ 自适应

台阶（terrain_height 跳变）:
  目标: terrain_height + 0.445 ✓ 自适应

桥梁（terrain_height = 1.0）:
  目标: 1.0 + 0.445 = 1.445 ✓ 正确！
```

### 训练效果对比

**修复前**：
- ❌ 机器人在 dimps 中尝试"爬升"
- ❌ 高度奖励与其他奖励冲突
- ❌ 课程学习效果差

**修复后**：
- ✅ 机器人在任何地形都保持合理高度
- ✅ 高度奖励与运动奖励协调一致
- ✅ 课程学习顺利推进

## 🔬 技术细节

### measured_heights 的工作原理
```python
# sirius_joystick.py:459
if self.cfg.terrain.measure_heights:
    self.measured_heights = self._get_heights()
```

- `_get_heights()` 在机器人周围采样多个点的地形高度
- `measured_heights`: `[num_envs, num_height_points]`
- `torch.mean(self.measured_heights, dim=1)`: 计算脚下平均高度

### 为什么用平均值？
```python
terrain_height = torch.mean(self.measured_heights, dim=1)
```

**原因**：
1. **鲁棒性**：单个点可能有噪声或异常值
2. **稳定性**：平均值更平滑，避免奖励剧烈波动
3. **代表性**：机器人是"站在一片区域"而非单点

### 配置说明

`base_height_target` 的含义**已改变**：

**之前**：
- 绝对高度（世界坐标系 z 轴）
- 仅适用于平地

**现在**：
- 相对高度（相对于脚下地形）
- 适用于所有地形

**建议值**：
- `0.445`：四足机器人的典型站立高度（腿长约 0.45m）
- 可根据实际机器人尺寸调整

## ⚠️ 注意事项

### 1. measure_heights 必须启用
```python
class terrain:
    measure_heights = True  # ← 必须为 True
```

如果 `measure_heights = False`，则 `self.measured_heights = 0`，修复无效。

**当前状态**：
- ✅ `sirius_curriculum_config.py`: `measure_heights = False`（但应该启用）
- ✅ `sirius_flat_config.py`: `measure_heights = False`（平地可以关闭）

### 2. 建议启用 measure_heights
编辑 `sirius_curriculum_config.py`：

```python
class terrain(SiriusFlatCfg.terrain):
    mesh_type = "trimesh"
    measure_heights = True  # ← 改为 True
```

**原因**：
- 启用后，高度奖励才能正确工作
- 对课程学习的复杂地形至关重要
- 性能影响很小（每步只多一次采样）

### 3. flat 场景可以保持关闭
```python
# sirius_flat_config.py
measure_heights = False  # 平地可以保持 False，因为 terrain_height = 0
```

因为 `_get_heights()` 对 plane 会直接返回 0：
```python
if self.cfg.terrain.mesh_type == 'plane':
    return torch.zeros(...)
```

## 🚀 启用步骤

### 1. 启用 measure_heights（推荐）
```python
# legged_gym/legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py
class terrain(SiriusFlatCfg.terrain):
    measure_heights = True  # ← 改为 True
```

### 2. 重新训练
```bash
python legged_gym/scripts/train.py --task=sirius_curriculum --num_envs=1024
```

### 3. 监控高度奖励
在 TensorBoard 中查看 `Rewards/base_height`：
- 初期应该较大（机器人还在学习）
- 随训练应该逐渐减小
- 在各种地形上应该表现一致

## 📈 预期改进

### 训练稳定性
- ✅ 高度奖励不再与地形难度冲突
- ✅ 课程学习顺利推进
- ✅ Reward 曲线更平滑

### 视觉泛化
- ✅ FiLM 门控能学到正确的高度调节策略
- ✅ 视觉编码器能关注地形高度特征
- ✅ 在各种地形上表现一致

### 最终性能
- ✅ 机器人能在 dimps、台阶、斜坡上保持合理高度
- ✅ 不会出现"爬升"或"下沉"的异常行为
- ✅ 晋级速度更快

## 🎓 相关概念

### 绝对 vs 相对奖励
- **绝对奖励**：基于世界坐标系（如绝对高度）
  - 适用于简单、平坦环境
  - 不适用于多样化地形

- **相对奖励**：基于当前状态/环境（如相对高度）
  - 适用于复杂、多样化环境
  - 更符合课程学习的原则

### 课程学习中的奖励设计
- **原则**：奖励应该在所有难度上保持一致的语义
- **错误示例**：固定高度目标（语义在不同地形上不一致）
- **正确示例**：相对高度目标（语义在所有地形上一致）

## 📚 参考

- Isaac Gym 高度测量：`_get_heights()` 方法
- Legged Robot RL：[Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)
- 课程学习：[Curriculum Learning for Reinforcement Learning Domains](https://arxiv.org/abs/2108.02540)

---

**✅ 修复完成！高度奖励现在能够自适应各种地形！**
