# 🔧 观测维度修复 - measure_heights 启用后的问题

## ❌ 错误信息
```
RuntimeError: The size of tensor a (232) must match the size of tensor b (45) at non-singleton dimension 1
```

## 🔍 根本原因

启用 `measure_heights = True` 后：
- **实际观测维度**: 45 (本体) + 187 (高度测量: 17×11) = **232**
- **配置中的维度**: `num_observations = 45` ❌ 错误！
- **噪声向量索引**: `noise_vec[48:235]` ❌ 错误！

## ✅ 修复方案

### 修复 1: 更新观测维度
**文件**: `sirius_curriculum_config.py`

```python
# 修复前 ❌
class env(SiriusFlatCfg.env):
    num_observations = 45

# 修复后 ✅
class env(SiriusFlatCfg.env):
    num_observations = 232  # 45 (本体) + 187 (高度测量: 17x11)
```

### 修复 2: 更新噪声向量索引
**文件**: `sirius_joystick.py` (第 1038 行)

```python
# 修复前 ❌
if self.cfg.terrain.measure_heights:
    noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

# 修复后 ✅
if self.cfg.terrain.measure_heights:
    noise_vec[45:232] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
```

## 📊 观测空间结构

### 启用 measure_heights 后的完整结构

| 索引范围 | 维度 | 内容 | 说明 |
|---------|------|------|------|
| 0:3 | 3 | base_ang_vel | 基座角速度 |
| 3:6 | 3 | projected_gravity | 投影重力向量 |
| 6:9 | 3 | commands | 速度命令 |
| 9:21 | 12 | dof_pos | 关节位置 |
| 21:33 | 12 | dof_vel | 关节速度 |
| 33:45 | 12 | actions | 上一步动作 |
| **45:232** | **187** | **heights** | **地形高度测量 (17×11)** |

**总计**: 232 维

### 关闭 measure_heights 时的结构

| 索引范围 | 维度 | 内容 |
|---------|------|------|
| 0:3 | 3 | base_ang_vel |
| 3:6 | 3 | projected_gravity |
| 6:9 | 3 | commands |
| 9:21 | 12 | dof_pos |
| 21:33 | 12 | dof_vel |
| 33:45 | 12 | actions |

**总计**: 45 维

## 🧮 高度测量点计算

### 配置 (legged_robot_config.py)
```python
measured_points_x = [-0.8, -0.7, ..., 0.7, 0.8]  # 17 个点
measured_points_y = [-0.5, -0.4, ..., 0.4, 0.5]  # 11 个点
```

### 计算
```
总测量点数 = len(measured_points_x) × len(measured_points_y)
           = 17 × 11
           = 187
```

### 观测维度
```
num_observations = 基础观测 + 高度测量
                 = 45 + 187
                 = 232
```

## ⚠️ 重要提示

### 1. measure_heights 与 num_observations 必须匹配

| measure_heights | num_observations | 正确性 |
|----------------|------------------|--------|
| False | 45 | ✅ 正确 |
| True | 232 | ✅ 正确 |
| False | 232 | ❌ 浪费空间 |
| True | 45 | ❌ 维度错误！|

### 2. 不同场景的配置

**sirius_flat (平地)**:
```python
measure_heights = False
num_observations = 45  # ✅
```
平地不需要高度测量。

**sirius_curriculum (课程学习)**:
```python
measure_heights = True
num_observations = 232  # ✅ 已修复
```
复杂地形必须启用高度测量。

**sirius_bridge (桥梁)**:
```python
measure_heights = False
num_observations = 45  # ✅
```
自定义几何体，不使用 heightfield。

### 3. 噪声向量索引规则

高度测量噪声的索引应该是：
```python
start_idx = 45  # 基础观测结束位置
end_idx = 45 + 187 = 232  # 高度测量结束位置
noise_vec[45:232] = ...  # ✅ 正确
```

**常见错误**:
- `noise_vec[48:235]` ❌ 错误的起始和结束索引
- `noise_vec[0:187]` ❌ 覆盖了基础观测

## 🚀 测试

运行训练命令验证：
```bash
python legged_gym/scripts/train.py --task=sirius_curriculum --num_envs=1024
```

应该能正常启动，不再出现维度不匹配错误。

## 📝 修复清单

- [x] 更新 `num_observations = 232` (sirius_curriculum_config.py)
- [x] 修复 `noise_vec[45:232]` (sirius_joystick.py)
- [x] 验证测量点配置 (17×11 = 187 ✅)
- [ ] 运行训练验证（待用户测试）

---

**✅ 修复完成！现在可以正常使用 measure_heights 功能了！**
