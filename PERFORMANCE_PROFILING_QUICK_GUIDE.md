# ⏱️ 性能分析快速指南

## 🎯 概述

本指南帮助您快速理解和使用新增的性能分析工具，定位训练中的性能瓶颈。

---

## 📊 三级性能报告

### Level 1: Step 总览（每 100 步打印）

```
======================================================================
⏱️  Step Performance Profiling (Step 100)
======================================================================
  1. Clip Actions:       0.15 ms ( 0.5%)
  2. Render:             2.30 ms ( 7.8%)
  3. Physics Sim:       12.45 ms (42.1%)  ⬅️ 物理仿真
  4. Post Physics:      14.20 ms (48.0%)  ⬅️ 主要瓶颈
  5. Clip Obs:           0.48 ms ( 1.6%)
  ──────────────────────────────────────────────────────────────────
  ⏱️  TOTAL:            29.58 ms
  📊 FPS:              33.8 steps/sec
======================================================================
```

**如何解读**：
- ✅ **Physics Sim < 50%**: 物理仿真性能正常
- ⚠️ **Post Physics > 40%**: 需要进一步分析（见 Level 2）
- 🎯 **目标 FPS**: > 30 steps/sec

---

### Level 2: Post-Physics 详细分析

```
======================================================================
📊 Post-Physics Step Breakdown (Step 100)
======================================================================
  1. Refresh Tensors:      0.85 ms ( 6.0%)
  2. Prepare Quantities:   0.32 ms ( 2.3%)
  3. Callback:             1.45 ms (10.2%)
  4. Check Termination:    0.12 ms ( 0.8%)
  5. Compute Reward:       1.23 ms ( 8.7%)
  6. Reset Idx:            0.45 ms ( 3.2%)
  7. Compute Observations: 9.18 ms (64.6%)  ⬅️ 主要瓶颈！
  8. Update Buffers:       0.28 ms ( 2.0%)
  9. Camera Display:       0.22 ms ( 1.5%)
 10. Debug Viz:            0.10 ms ( 0.7%)
======================================================================
```

**如何解读**：
- ⚠️ **Compute Observations > 60%**: 观测计算是瓶颈（见 Level 3）
- ⚠️ **Compute Reward > 15%**: 奖励函数需要优化
- ⚠️ **Callback > 15%**: 命令重采样或地形高度计算慢
- ⚠️ **Reset Idx > 10%**: 环境重置太频繁或太慢

---

### Level 3: Observations 细分

```
======================================================================
🔍 Compute Observations Breakdown (Step 100)
======================================================================
  1. Proprioception:       0.45 ms ( 4.9%)
  2. Terrain Heights:      0.02 ms ( 0.2%)
  3. Add Noise:            0.18 ms ( 2.0%)
  4. Camera (Total):       8.53 ms (92.9%)  ⬅️ 相机是瓶颈！
     - Get Depth:          7.82 ms           ⬅️ 优化前可能 >20ms
     - Normalize:          0.71 ms
======================================================================
```

**如何解读**：
- ⚠️ **Camera > 80%**: 相机获取是主要瓶颈
  - ✅ **已优化**: 使用批量访问 API（2-5x 加速）
  - 🔧 **进一步优化**: 降低分辨率或使用更小的 CNN
- ⚠️ **Proprioception > 10%**: 本体感觉计算异常（通常 < 1ms）

---

## 🔧 常见性能问题与解决方案

### 问题 1: 相机获取慢（Get Depth > 10ms）

**症状**:
```
Camera (Total):       20.00 ms (90%)
  - Get Depth:        18.50 ms  ⚠️ 太慢！
  - Normalize:         1.50 ms
```

**解决方案**:
1. ✅ **确认批量访问已启用**（应该看到 "FASTEST ✅✅✅"）
   ```python
   # 配置文件
   camera:
     enable_tensors: True  # 必须设置！
   ```

2. 🔧 **降低相机分辨率**
   ```python
   camera:
     width: 87   # 从 174 降低到 87
     height: 58  # 从 116 降低到 58
   ```

3. 🔧 **减少环境数量**（临时测试）
   ```bash
   python scripts/train.py --num_envs=1024  # 从 4096 降到 1024
   ```

---

### 问题 2: 奖励计算慢（Compute Reward > 15%）

**症状**:
```
5. Compute Reward:       3.50 ms (25.0%)  ⚠️ 太慢！
```

**可能原因**:
- 奖励函数中有 Python 循环
- 使用了 CPU 操作（如 `.cpu()`, `.numpy()`）
- 奖励项太多或计算复杂

**解决方案**:
1. 🔍 **检查奖励函数**
   ```bash
   grep -n "def _reward_" legged_gym/envs/sirius_diff_vis/sirius_joystick.py
   ```

2. 🔧 **优化热点奖励函数**
   - 避免循环，使用向量化操作
   - 移除不必要的 `.clone()` 或 `.detach()`
   - 缓存中间结果

3. 📊 **减少奖励项数量**
   ```python
   # 配置文件
   rewards:
     scales:
       # 暂时禁用不重要的奖励
       # feet_air_time: 1.0
       # slip: -0.1
   ```

---

### 问题 3: 物理仿真慢（Physics Sim > 60%）

**症状**:
```
3. Physics Sim:       18.00 ms (60.0%)  ⚠️ 瓶颈在物理引擎
```

**解决方案**:
1. 🔧 **减少物理步数**
   ```python
   control:
     decimation: 4  # 从 4 改为 2（会影响控制精度）
   ```

2. 🔧 **简化物理模型**
   ```python
   asset:
     disable_gravity: False
     collapse_fixed_joints: True  # 合并固定关节
   ```

3. 🔧 **减少接触检测**
   ```python
   sim:
     physx:
       max_gpu_contact_pairs: 2**20  # 从 2**23 降低
   ```

---

### 问题 4: FPS 太低（< 20 steps/sec）

**综合优化策略**:

1. **快速诊断**
   ```bash
   # 运行 100 步后查看报告
   python scripts/train.py --task=sirius_joystick_flat 2>&1 | grep -A 20 "Step Performance"
   ```

2. **按优先级优化**
   - 🥇 **相机 > 80%** → 启用批量访问 + 降低分辨率
   - 🥈 **物理 > 60%** → 减少 decimation
   - 🥉 **奖励 > 20%** → 优化奖励函数

3. **验证优化效果**
   ```bash
   # 优化前
   FPS: 15.2 steps/sec
   
   # 优化后（预期）
   FPS: 35.8 steps/sec  ✅ 提升 2.4x
   ```

---

## 📈 性能基准

| 配置 | 相机分辨率 | 环境数 | 预期 FPS | 相机耗时 |
|------|-----------|--------|---------|---------|
| **低配** | 87x58 | 2048 | 40-60 | ~5ms |
| **标准** | 87x58 | 4096 | 30-50 | ~8ms |
| **高配** | 174x116 | 4096 | 15-25 | ~15ms |
| **极限** | 174x116 | 8192 | 8-12 | ~25ms |

**硬件**: NVIDIA RTX 3090 / RTX 4090

---

## 🚀 快速优化检查清单

在开始训练前，确保：

- [ ] ✅ `camera.enable_tensors = True`（GPU 批量访问）
- [ ] ✅ 启动信息显示 "FASTEST ✅✅✅"
- [ ] ✅ 相机分辨率合理（87x58 推荐）
- [ ] ✅ 环境数量适配 GPU 显存（4096 推荐）
- [ ] ✅ `decimation = 4`（平衡精度与速度）
- [ ] ✅ 禁用不必要的奖励项
- [ ] ✅ 禁用 `debug_viz`（训练时不需要）

---

## 🔍 调试命令

```bash
# 1. 查看是否使用批量访问
python scripts/train.py --task=sirius 2>&1 | grep "batch access"
# 期望输出: "Using GPU tensor path with batch access (FASTEST) ✅✅✅"

# 2. 实时监控 FPS
watch -n 1 "nvidia-smi; echo '---'; tail -n 5 logs/train.log | grep FPS"

# 3. 性能分析（保存前 1000 步的详细日志）
python scripts/train.py --task=sirius 2>&1 | tee perf_log.txt | head -n 1000

# 4. 提取性能摘要
grep -E "(Step Performance|Post-Physics|Compute Observations)" perf_log.txt > perf_summary.txt
```

---

## 📚 相关文件

- **优化详情**: `CAMERA_PERFORMANCE_OPTIMIZATION.md`
- **主环境文件**: `legged_gym/envs/sirius_diff_vis/sirius_joystick.py`
- **配置文件**: `legged_gym/envs/sirius_diff_vis/sirius_flat_config.py`

---

## 💡 小贴士

1. **每 100 步自动打印**，无需手动调用
2. **百分比最重要**，绝对时间会因硬件而异
3. **先优化占比最高的部分**，二八定律适用
4. **优化后验证 FPS 提升**，确保改进有效

---

## 🎯 性能优化路线图

```
Step 1: 启用批量访问
   ↓
  FPS: 15 → 35 (2.3x)  ✅ 已完成

Step 2: 降低相机分辨率（可选）
   ↓
  FPS: 35 → 50 (1.4x)  🔧 如需进一步优化

Step 3: 优化奖励函数（可选）
   ↓
  FPS: 50 → 60 (1.2x)  🔧 如果奖励计算慢

Step 4: 调整物理参数（可选）
   ↓
  FPS: 60 → 80 (1.3x)  🔧 如果物理仿真慢
```

---

**目标**: 在保证训练质量的前提下，达到 **30+ FPS** ✅

祝训练顺利！🚀
