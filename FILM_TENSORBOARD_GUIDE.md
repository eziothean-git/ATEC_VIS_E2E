# FiLM 门控模块 - TensorBoard 监控指南

## 📊 概述

FiLM (Feature-wise Linear Modulation) 门控模块的 TensorBoard 日志已集成完毕。训练过程中会自动记录 scale 和 shift 的统计信息，帮助监控门控模块的学习过程。

## 🎯 监控的指标

### Scale 统计
- **FiLM/scale_mean**: scale 的均值
  - 期望：训练初期 ≈ 0，随训练逐步增长
  - 正常范围：-0.05 ~ 0.05（受 `film_scale_limit=0.1` 限制）

- **FiLM/scale_std**: scale 的标准差
  - 反映 scale 在不同维度的差异性
  - 标准差较大 = 不同本体特征受到不同强度的调制

- **FiLM/scale_min / scale_max**: scale 的最小值和最大值
  - 监控是否触及限幅边界 (±0.1)
  - 如果经常达到边界，考虑增大 `film_scale_limit`

- **FiLM/scale_abs_mean**: scale 绝对值的均值
  - 衡量 scale 的"激活程度"
  - 训练初期 ≈ 0，逐步增长表示门控在学习

### Shift 统计
- **FiLM/shift_mean**: shift 的均值
  - 期望：训练初期 ≈ 0，随训练逐步变化
  - 无限制，可以任意正负

- **FiLM/shift_std**: shift 的标准差
  - 反映 shift 在不同维度的差异性

- **FiLM/shift_min / shift_max**: shift 的最小值和最大值
  - 监控 shift 的范围

- **FiLM/shift_abs_mean**: shift 绝对值的均值
  - 衡量 shift 的"激活程度"

### 调制效果
- **FiLM/modulation_magnitude**: 调制幅度
  - 计算：`|modulated - proprio|` 的平均值
  - 衡量 FiLM 对本体特征的实际影响程度
  - 期望：训练初期 ≈ 0，随训练逐步增长

- **FiLM/relative_change**: 相对变化
  - 计算：`modulation_magnitude / mean(|proprio|)`
  - 百分比形式，更直观
  - 正常范围：1% ~ 20%

### 饱和度检测
- **FiLM/scale_saturation**: scale 饱和度
  - 计算：`tanh(scale_raw)` 中绝对值 > 0.9 的比例
  - 如果持续 > 50%，说明限幅可能过严，需要增大 `film_scale_limit`

## 📈 训练阶段的预期变化

### 阶段 1: 初期（0-100 iterations）
```
scale_abs_mean:    0.000000 → 0.001000
shift_abs_mean:    0.000000 → 0.005000
modulation_magnitude: 0.0 → 0.01
relative_change:      0% → 2%
```
- FiLM 从零初始化开始，逐步"苏醒"
- 此时影响很小，主要依靠纯 concat 的特征

### 阶段 2: 成长（100-500 iterations）
```
scale_abs_mean:    0.001000 → 0.010000
shift_abs_mean:    0.005000 → 0.020000
modulation_magnitude: 0.01 → 0.05
relative_change:      2% → 10%
```
- FiLM 开始学习有意义的门控策略
- 调制效果逐步增强

### 阶段 3: 稳定（500+ iterations）
```
scale_abs_mean:    ~0.010 - 0.030
shift_abs_mean:    ~0.020 - 0.050
modulation_magnitude: ~0.05 - 0.10
relative_change:      ~10% - 20%
```
- FiLM 达到稳定状态
- 继续微调以适应不同场景

## 🔍 如何使用 TensorBoard

### 启动 TensorBoard
```bash
# 在项目根目录运行
tensorboard --logdir=logs/

# 或指定特定实验
tensorboard --logdir=logs/sirius_flat_[timestamp]
```

### 查看 FiLM 指标
1. 在浏览器中打开 TensorBoard (通常是 http://localhost:6006)
2. 点击 "SCALARS" 标签
3. 在左侧搜索栏输入 "FiLM" 即可看到所有相关指标
4. 建议同时查看的指标组合：
   - `FiLM/scale_abs_mean` 和 `FiLM/shift_abs_mean` (监控激活程度)
   - `FiLM/modulation_magnitude` 和 `Train/mean_reward` (调制效果 vs 性能)
   - `FiLM/scale_saturation` (检查限幅)

### 调试建议

#### 问题 1: FiLM 不学习（scale 和 shift 一直接近 0）
**症状**：
- `scale_abs_mean` 和 `shift_abs_mean` 在数百次迭代后仍然 < 0.001
- `modulation_magnitude` 接近 0

**可能原因**：
1. 学习率过低，门控网络学不动
2. 视觉特征质量差，无法提供有效信号
3. 门控网络容量不足

**解决方案**：
1. 检查整体学习率是否正常
2. 查看 `Loss/value_function` 和 `Loss/surrogate` 是否下降
3. 尝试增加 `film_hidden_dims` (如 `[128]` 或 `[64, 64]`)
4. 检查深度图数据是否正常（不全是 0 或 NaN）

#### 问题 2: FiLM 过强导致训练不稳定
**症状**：
- `modulation_magnitude` 迅速增长 > 0.5
- `relative_change` > 50%
- 训练 reward 剧烈震荡

**可能原因**：
1. `film_scale_limit` 设置过大
2. 没有正确的零初始化

**解决方案**：
1. 减小 `film_scale_limit` (从 0.1 降到 0.05)
2. 检查代码中是否有 `nn.init.zeros_()` 初始化
3. 降低整体学习率

#### 问题 3: FiLM 达到饱和
**症状**：
- `FiLM/scale_saturation` > 50%
- `scale_min` 和 `scale_max` 长期卡在边界

**解决方案**：
1. 增大 `film_scale_limit` (从 0.1 增到 0.2)
2. 或者接受饱和（说明网络认为需要更强的调制）

## 🎨 TensorBoard 可视化示例

### 理想的训练曲线

```
scale_abs_mean
    ^
0.03|                          ___________
    |                     ____/
0.02|                ____/
    |           ____/
0.01|      ____/
    | ____/
0.00|/____________________________________> iterations
    0   200   400   600   800  1000

modulation_magnitude vs mean_reward
    ^
 10 |     (reward)         ___________
    |                 ____/
  8 |            ____/
    |       ____/      (modulation)
  6 |  ____/    ___________
    | /    ____/
  4 |_____/
    |
  2 |___________________________________> iterations
    0   200   400   600   800  1000
```

两者应该呈正相关：FiLM 调制增强 → 性能提升

## 📝 控制台输出

训练时每 50 次迭代会在控制台打印 FiLM 统计：

```
[FiLM Stats @ iter 100]
  scale: mean=0.001234, std=0.003456, range=[-0.012345, 0.010987]
  shift: mean=-0.002345, std=0.012345, range=[-0.045678, 0.038901]
  modulation: magnitude=0.015678, relative=3.14%
```

## ⚙️ 配置参数调整

如果需要调整 FiLM 行为，修改 `sirius_shared_model.py`:

```python
class policy(LeggedRobotCfgPPO.policy):
    # ... 其他配置 ...
    
    # FiLM 配置
    use_film_gating = True       # 是否启用
    film_hidden_dims = [64]      # 门控 MLP 隐藏层（增大提升容量）
    film_activation = 'elu'      # 激活函数
    film_scale_limit = 0.1       # scale 限幅（根据饱和度调整）
    film_scale_init = 0.0        # 保持 0.0
    film_shift_init = 0.0        # 保持 0.0
```

## 🚀 最佳实践

1. **训练前**：
   - 确认 `use_film_gating = True`
   - 使用默认的 `film_scale_limit = 0.1`
   - 确保零初始化代码存在

2. **训练中**：
   - 定期查看 TensorBoard 的 FiLM 指标
   - 前 100 次迭代：验证 scale/shift 从 0 开始增长
   - 200-500 次迭代：观察调制效果与性能的关系
   - 关注 `scale_saturation`，如果 > 50% 考虑调整

3. **训练后**：
   - 对比启用/禁用 FiLM 的性能差异
   - 分析哪些本体特征受 FiLM 调制最多
   - 考虑可视化 scale 和 shift 的维度分布

## 📞 问题排查

如果 TensorBoard 没有显示 FiLM 指标：

1. 检查是否启用了 FiLM：
   ```python
   # 在训练日志中应该看到：
   [FiLM Gating Module] Enabled
   ```

2. 检查 `_log_film_statistics` 是否被调用：
   - 在 `on_policy_runner.py` 的 `log` 方法中搜索 `_log_film_statistics`

3. 查看控制台是否有错误信息：
   ```
   [Warning] Failed to log FiLM statistics: ...
   ```

4. 确认使用的是视觉策略 `VisionProprioceptionActorCritic`

## 🎯 总结

- ✅ FiLM 统计已自动记录到 TensorBoard
- ✅ 涵盖 13 个监控指标
- ✅ 每 50 次迭代打印控制台摘要
- ✅ 异常时静默失败，不影响训练
- ✅ 三个场景（平地、课程、桥梁）统一生效

训练时只需正常运行，TensorBoard 会自动记录所有 FiLM 相关信息！
