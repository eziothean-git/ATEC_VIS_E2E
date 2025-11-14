# 地形类型跳过修复 (Terrain Type Skip Fix)

## 🐛 问题描述

**错误理解**：之前一直在跳过**地形难度级别（行）**的第3-4级，而不是**地形类型（列）**的第3-4种。

### 地形网格结构
```
terrain_origins[行, 列] = terrain_origins[难度级别, 地形类型]

行（Rows）：难度级别（0-9）
  - 0-1: 低难度（小起伏）
  - 2-4: 中等难度
  - 5-7: 高难度
  - 8-9: 极限难度

列（Cols）：地形类型（0-7）
  - 0-1: 斜坡（上坡/下坡）          ✅ 简单
  - 2:   斜坡 + 粗糙表面             ✅ 中等
  - 3-4: 台阶（上升/下降）          ❌ 需要跳过！
  - 5:   离散障碍物                 ✅ 复杂
  - 6:   踏脚石                     ✅ 困难
  - 7:   缝隙/坑洞                  ✅ 最困难
```

### 为什么要跳过台阶类型（Col 3-4）？
1. **深坑导致策略欺骗**：机器人可能学会"趴下"通过台阶而不是正常行走
2. **视觉失效**：深度相机在大落差处容易失效
3. **真实场景不常见**：实际应用中很少遇到规整的台阶地形

## ✅ 修复方案

### 1. 配置文件修改

**文件**: `sirius_curriculum_config.py`

```python
# 修改前（错误）
skip_terrain_levels = [3, 4]  # 跳过难度3-4（这是行，不是列！）

# 修改后（正确）
skip_terrain_types = [3, 4]   # 跳过地形类型3-4（台阶上下）
```

### 2. 地形类型分配逻辑

**文件**: `sirius_joystick.py` → `_get_env_origins()`

**修改前**：均匀分配到8种地形类型
```python
self.terrain_types = torch.div(
    torch.arange(self.num_envs), 
    (self.num_envs/self.cfg.terrain.num_cols), 
    rounding_mode='floor'
).to(torch.long)
# 结果：每种类型 num_envs/8 个环境（完全均匀）
```

**修改后**：按比例随机分配，跳过类型3-4
```python
skip_types = getattr(self.cfg.terrain, 'skip_terrain_types', [])
available_types = [i for i in range(num_cols) if i not in skip_types]

# 根据 terrain_proportions 按比例分配
original_proportions = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
filtered_proportions = [original_proportions[i] for i in available_types]
filtered_proportions /= sum(filtered_proportions)  # 重新归一化

self.terrain_types = random_choice(available_types, p=filtered_proportions)
```

### 3. 地形难度分配逻辑

**保持不变**：难度级别（行）按课程学习策略分配
- 80% 环境：难度 0-1（简单）
- 15% 环境：难度 2（中等）
- 5% 环境：视觉探索（所有难度随机）

### 4. 移除错误的跳过逻辑

**文件**: `sirius_joystick.py` → `_update_terrain_curriculum()`

移除了之前在难度级别更新时跳过3-4的逻辑（那是错误的）。现在：
- **难度级别（行）**：可以自由增长 0→9
- **地形类型（列）**：固定分配，不包含类型3-4

## 📊 预期结果

### 地形类型分布（列）
```
类型 0 (斜坡上):      ~102 个环境 (20.0%)
类型 1 (斜坡下):      ~102 个环境 (20.0%)
类型 2 (粗糙斜坡):    ~77  个环境 (15.0%)
类型 3 (台阶上):      0   个环境 (0.0%)    🚫 已跳过
类型 4 (台阶下):      0   个环境 (0.0%)    🚫 已跳过
类型 5 (障碍物):      ~77  个环境 (15.0%)
类型 6 (踏脚石):      ~77  个环境 (15.0%)
类型 7 (缝隙):        ~77  个环境 (15.0%)
```

### 地形难度分布（行）
```
难度 0:  ~205 个环境 (40%)  - 简单
难度 1:  ~205 个环境 (40%)  - 简单
难度 2:  ~77  个环境 (15%)  - 中等
难度 3:  ~6   个环境 (1%)   - 视觉探索
难度 4:  ~6   个环境 (1%)   - 视觉探索
难度 5:  ~4   个环境 (1%)   - 视觉探索
难度 6:  ~4   个环境 (1%)   - 视觉探索
难度 7:  ~3   个环境 (0.5%) - 视觉探索
难度 8:  ~1   个环境 (0.2%) - 视觉探索
难度 9:  ~1   个环境 (0.2%) - 视觉探索
```

## 🧪 验证方法

### 运行训练并检查初始化日志
```bash
python scripts/train.py --task=sirius_curriculum --num_envs=512 --headless
```

查找输出：
```
✅ 使用地形类型比例配置（跳过类型 [3, 4]）:
  类型 0: XXX 个环境 (预期 25.0%, 实际 XX.X%)
  类型 1: XXX 个环境 (预期 25.0%, 实际 XX.X%)
  类型 2: XXX 个环境 (预期 18.8%, 实际 XX.X%)
  🚫 类型 3: 已跳过（台阶地形）
  🚫 类型 4: 已跳过（台阶地形）
  类型 5: XXX 个环境 (预期 12.5%, 实际 XX.X%)
  类型 6: XXX 个环境 (预期 12.5%, 实际 XX.X%)
  类型 7: XXX 个环境 (预期 6.3%, 实际 XX.X%)
```

### 检查错误
如果看到以下输出，说明还有问题：
```
❌ 错误：类型 3 有 XX 个环境（应该被跳过）
❌ 错误：类型 4 有 XX 个环境（应该被跳过）
```

## 📝 修改文件清单

1. ✅ `sirius_curriculum_config.py` (第106行)
   - `skip_terrain_levels` → `skip_terrain_types`

2. ✅ `sirius_joystick.py` (第1740-1790行)
   - 移除难度级别的跳过逻辑
   - 添加地形类型的跳过逻辑

3. ✅ `sirius_joystick.py` (第1810-1850行)
   - 修改地形类型分配：从均匀分配改为按比例随机分配
   - 添加跳过类型3-4的逻辑

4. ✅ `sirius_joystick.py` (第905-960行)
   - 简化 `_update_terrain_curriculum`
   - 移除错误的跳过难度级别逻辑

## 🎯 训练建议

1. **启动训练**：
   ```bash
   python scripts/train.py --task=sirius_curriculum --num_envs=512 --headless
   ```

2. **监控 TensorBoard**：
   - 检查 `Terrain/type_X` 指标：类型3和4应该始终为0
   - 检查 `Terrain/level_X` 指标：所有难度级别都应该有环境

3. **观察训练稳定性**：
   - 没有台阶地形后，训练应该更稳定
   - 不会出现机器人"趴下"的策略欺骗行为

## ⚠️ 注意事项

- **地形类型（列）是固定的**：每个环境在初始化后就确定了地形类型，不会改变
- **难度级别（行）会动态变化**：根据课程学习策略自动升降级
- **跳过的类型永远不会出现**：初始化时就排除了，不需要在更新时检查
