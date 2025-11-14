# Bug修复: measured_heights 类型错误

## 🐛 问题描述

在运行 `python train.py --task=sirius --headless` 时遇到以下错误：

```
TypeError: mean() received an invalid combination of arguments - got (int, dim=int)
```

错误发生在 `sirius_joystick.py::_reward_base_height()` 函数中。

## 🔍 根本原因

1. **初始化问题**: `self.measured_heights` 在 `_post_physics_step_callback()` 中被初始化为整数 `0`：
   ```python
   self.measured_heights = 0  # ❌ 错误：整数而非 tensor
   ```

2. **条件检查不足**: `_reward_base_height()` 函数假设 `measured_heights` 是 tensor，直接调用 `torch.mean()`：
   ```python
   terrain_height = torch.mean(self.measured_heights, dim=1)  # ❌ 当 measured_heights=0 时报错
   ```

3. **环境差异**: 
   - `task=sirius` (平地): `measure_heights = False`，不会调用 `_get_heights()` 更新 `measured_heights`
   - `task=sirius_curriculum` (地形): `measure_heights = True`，会更新为有效 tensor

## ✅ 修复方案

### 修复1: 初始化为 None
```python
# 文件: sirius_joystick.py, 行 1477
self.measured_heights = None  # ✅ 初始化为 None，在有地形测量时会被更新为 tensor
```

### 修复2: 添加类型检查
```python
# 文件: sirius_joystick.py, _reward_base_height()
def _reward_base_height(self):
    # 如果启用了地形高度测量，使用相对高度
    if hasattr(self, 'measured_heights') and self.measured_heights is not None:
        terrain_height = torch.mean(self.measured_heights, dim=1)  # [num_envs]
        base_height_above_terrain = self.root_states[:, 2] - terrain_height
    else:
        # 平地环境：直接使用绝对高度（假设地面在 z=0）
        base_height_above_terrain = self.root_states[:, 2]
    
    return torch.square(base_height_above_terrain - self.cfg.rewards.base_height_target)
```

## 🎯 影响范围

### 修复前
- ❌ `task=sirius` (平地) 无法运行
- ✅ `task=sirius_curriculum` (地形) 可以运行

### 修复后
- ✅ `task=sirius` (平地) 可以运行
- ✅ `task=sirius_curriculum` (地形) 可以运行
- ✅ 两种环境都能正确计算 base_height 奖励

## 📝 测试建议

### 测试1: 平地环境
```bash
python train.py --task=sirius --headless --max_iterations=10
```

**预期结果**:
- 不报 TypeError
- base_height 奖励使用绝对高度计算
- 训练正常进行

### 测试2: Curriculum环境
```bash
python train.py --task=sirius_curriculum --headless --max_iterations=10
```

**预期结果**:
- 不报 TypeError
- base_height 奖励使用相对高度计算（减去地形高度）
- 训练正常进行

### 测试3: 验证奖励逻辑
在第一个episode结束后，检查 tensorboard:
```bash
tensorboard --logdir logs/
```

查看 `Train/base_height` 奖励是否合理。

## 🔄 后续优化建议

1. **统一初始化**: 考虑在 `__init__` 中统一初始化所有 buffer，而不是在 `_post_physics_step_callback` 中
2. **类型注解**: 添加类型注解明确 `measured_heights` 的类型
3. **单元测试**: 为奖励函数添加单元测试，覆盖有/无地形测量的情况

## 📚 相关文件

- `legged_gym/envs/sirius_diff_vis/sirius_joystick.py`
  - 行 1477: 初始化 `measured_heights`
  - 行 2000-2013: `_reward_base_height()` 函数
  
- `legged_gym/envs/sirius_diff_vis/sirius_flat_config.py`
  - `terrain.measure_heights = False` (平地)
  
- `legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py`
  - `terrain.measure_heights = True` (地形课程)

---

**修复日期**: 2025年11月14日
**修复人员**: AI Assistant
**测试状态**: ⏳ 待测试
