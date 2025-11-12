# FiLM 门控模块 - 快速参考

## 🚀 快速开始

### 训练（默认启用 FiLM）
```bash
python train.py --task=sirius_flat --num_envs=1024
```

### 查看 TensorBoard
```bash
tensorboard --logdir=logs/
# 访问: http://localhost:6006
# 查看 "FiLM" 标签页
```

## 📊 关键监控指标

| 指标 | 初始值 | 稳定值 | 说明 |
|------|--------|--------|------|
| `FiLM/scale_abs_mean` | ~0.000 | 0.01-0.03 | scale 激活程度 |
| `FiLM/shift_abs_mean` | ~0.000 | 0.02-0.05 | shift 激活程度 |
| `FiLM/modulation_magnitude` | ~0.000 | 0.05-0.10 | 调制幅度 |
| `FiLM/relative_change` | ~0% | 10-20% | 相对变化 |
| `FiLM/scale_saturation` | ~0% | <50% | 饱和度（避免过高） |

## ⚙️ 配置参数 (`sirius_shared_model.py`)

```python
class policy(LeggedRobotCfgPPO.policy):
    use_film_gating = True       # 启用/禁用
    film_hidden_dims = [64]      # 门控 MLP 隐藏层
    film_activation = 'elu'      # 激活函数
    film_scale_limit = 0.1       # scale 限幅 (±0.1)
    film_scale_init = 0.0        # 保持 0.0
    film_shift_init = 0.0        # 保持 0.0
```

## 🔧 常见问题速查

### FiLM 不学习？
**症状**: scale_abs_mean 持续 < 0.001  
**方案**: 
- 增加 `film_hidden_dims = [128]`
- 检查视觉编码器
- 确认学习率正常

### FiLM 过强？
**症状**: modulation_magnitude > 0.5, reward 震荡  
**方案**:
- 减小 `film_scale_limit = 0.05`
- 降低学习率

### 达到饱和？
**症状**: scale_saturation > 50%  
**方案**:
- 增大 `film_scale_limit = 0.2`
- 或接受饱和

## 📈 训练阶段预期

```
迭代 0-100:   scale ≈ 0.001, shift ≈ 0.005, 调制 ≈ 2%
迭代 100-500: scale ≈ 0.010, shift ≈ 0.020, 调制 ≈ 10%
迭代 500+:    scale ≈ 0.020, shift ≈ 0.040, 调制 ≈ 15%
```

## 🎯 成功标志

- ✅ scale/shift 从 0 逐步增长
- ✅ modulation_magnitude 稳定在 0.05-0.10
- ✅ Train/mean_reward 持续提升
- ✅ scale_saturation < 50%
- ✅ 控制台显示 `[FiLM Stats]`

## 📝 核心公式

```python
# FiLM 调制
modulated_proprio = proprio * (1 + scale) + shift

# scale 限幅
scale = film_scale_limit * tanh(scale_raw)  # ∈ [-0.1, 0.1]
```

## 🔗 详细文档

- `FILM_TENSORBOARD_GUIDE.md` - 完整监控指南
- `FILM_INTEGRATION_SUMMARY.md` - 集成总结
- TensorBoard: http://localhost:6006

## ⚡ 一键测试

```bash
# 测试核心逻辑
conda run -n sirius2 python test_film_core.py

# 测试 TensorBoard 日志
conda run -n sirius2 python test_film_tensorboard.py
```

## 🎨 架构图

```
深度图 → CNN → 视觉特征 (32)
                  ↓
               FiLM MLP
                  ↓
         [scale (45) | shift (45)]
                  ↓
本体 (45) → FiLM调制 → 调制后本体 (45)
                  ↓ concat
           融合特征 (77)
                  ↓
         Actor/Critic MLP
                  ↓
           动作 / 值函数
```

---

**快速联系**: 查看 `FILM_INTEGRATION_SUMMARY.md` 获取完整信息
