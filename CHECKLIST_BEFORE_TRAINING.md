# ✅ 训练前性能优化检查清单

## 🎯 使用说明
在开始训练前，逐项检查以下配置，确保所有优化都已启用。

---

## 📋 必检项（Critical）

### 1️⃣ 相机批量访问 - 最关键！🔴
**文件**: `legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py`

```python
class camera:
    enable_tensors = True  # ⚠️ 必须为 True！
```

**验证方法**:
```bash
# 运行训练，查看启动日志
python scripts/train.py --task=sirius_curriculum 2>&1 | grep "batch access"

# 应该看到（✅ 正确）:
[Camera Debug] Using GPU tensor path with batch access (FASTEST) ✅✅✅

# 不应该看到（❌ 错误）:
[Camera Debug] Using CPU/NumPy path (SLOW) ⚠️
```

**影响**: 2-5x 相机性能提升

---

### 2️⃣ Heightfield 地形 - 很重要！🟠
**文件**: `legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py`

```python
class terrain:
    mesh_type = "heightfield"  # ⚠️ 不是 "trimesh"！
    curriculum = True
```

**验证方法**:
```bash
# 检查配置文件
grep "mesh_type" legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py

# 应该看到（✅ 正确）:
mesh_type = "heightfield"

# 不应该看到（❌ 错误）:
mesh_type = "trimesh"
```

**影响**: 3-5x 物理仿真加速

---

### 3️⃣ GPU 设备检查 🟡
**验证方法**:
```bash
# 检查 CUDA 是否可用
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 应该看到（✅ 正确）:
CUDA: True
GPU: NVIDIA GeForce RTX 3090  # 或其他 GPU 名称

# 不应该看到（❌ 错误）:
CUDA: False
```

**影响**: 无 GPU = 无法训练

---

## 📋 推荐项（Recommended）

### 4️⃣ 相机分辨率
```python
class camera:
    width = 87   # 推荐：87（平衡精度和速度）
    height = 58  # 推荐：58
```

**调整建议**:
- 🐢 如果太慢 (FPS < 30): 降到 58x39
- 🚀 如果够快 (FPS > 60): 保持 87x58 或升到 174x116

---

### 5️⃣ 环境数量
```python
class env:
    num_envs = 4096  # 推荐：4096（RTX 3090/4090）
```

**调整建议**:
- GPU 显存 < 16GB: 2048
- GPU 显存 = 24GB: 4096
- GPU 显存 > 40GB: 8192

---

### 6️⃣ 调试可视化（训练时应禁用）
```python
# sirius_joystick.py
self.debug_viz = False  # ⚠️ 训练时设为 False
```

**验证方法**:
```bash
# 启动时不要使用 --render
python scripts/train.py --task=sirius_curriculum  # ✅ 正确（headless）
python scripts/train.py --task=sirius_curriculum --render  # ❌ 慢（有 viewer）
```

---

## 📋 可选项（Optional）

### 7️⃣ 物理仿真精度
```python
class control:
    decimation = 4  # 默认：4（平衡精度和速度）
```

**调整建议**:
- 追求速度: decimation = 2（但控制精度降低）
- 追求精度: decimation = 8（但速度变慢）

---

### 8️⃣ PPO 超参数
```python
class algorithm:
    num_learning_epochs = 8   # 默认：8
    num_mini_batches = 2      # 默认：2
```

**调整建议**:
- GPU 利用率低 (<50%): 增加 num_mini_batches 到 4
- GPU 显存不足: 减少 num_mini_batches 到 1

---

## 🚀 启动前最后检查

### 一键检查脚本
```bash
# 创建检查脚本
cat > check_config.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "⏱️  Performance Optimization Checklist"
echo "=========================================="
echo ""

# 1. Camera batch access
echo "1️⃣  Camera batch access:"
grep -A 1 "class camera" legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py | grep "enable_tensors"
echo ""

# 2. Heightfield terrain
echo "2️⃣  Terrain type:"
grep "mesh_type" legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py | grep -v "#"
echo ""

# 3. CUDA check
echo "3️⃣  CUDA availability:"
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo ""

# 4. Camera resolution
echo "4️⃣  Camera resolution:"
grep -E "(width|height) =" legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py | grep -v "#" | head -2
echo ""

# 5. Num envs
echo "5️⃣  Number of environments:"
grep "num_envs" legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py | grep -v "#" | head -1
echo ""

echo "=========================================="
echo "✅ Checklist complete!"
echo "=========================================="
EOF

# 运行检查
bash check_config.sh
```

---

## 📊 预期性能指标

训练启动后，查看前 100 步的性能报告，确保达到以下指标：

| 指标 | 目标 | 异常阈值 | 解决方案 |
|------|------|---------|---------|
| **总 FPS** | > 45 | < 30 | 检查项 1、2 |
| **Physics Sim** | 8-15ms | > 20ms | 检查项 2（heightfield） |
| **Camera Get** | 5-10ms | > 15ms | 检查项 1（batch access） |
| **Compute Obs** | 10-15ms | > 25ms | 检查项 1、4（分辨率） |
| **GPU 显存** | 1.5-2GB | > 3GB | 检查项 4、5（分辨率/envs） |

---

## 🎯 快速启动流程

```bash
# Step 1: 运行检查清单
bash check_config.sh

# Step 2: 启动训练
python scripts/train.py --task=sirius_curriculum --num_envs=4096

# Step 3: 查看前 100 步的性能报告（大约 2-3 分钟后）
# 报告会自动打印

# Step 4: 验证关键指标
# - FPS > 45 ✅
# - Physics Sim < 15ms ✅
# - Camera Get < 10ms ✅

# Step 5: 如果指标正常，继续训练！🎉
```

---

## ⚠️ 常见错误

### 错误 1: enable_tensors = False
```
[Camera Debug] Using CPU/NumPy path (SLOW) ⚠️
FPS: 15-20  # 太慢！
```
**修复**: 改为 `enable_tensors = True`

### 错误 2: mesh_type = "trimesh"
```
Physics Sim: 35-45ms  # 太慢！
FPS: 15-25
```
**修复**: 改为 `mesh_type = "heightfield"`

### 错误 3: 使用了 viewer（调试模式）
```
命令: python train.py --task=sirius_curriculum --render
Render: 15-25ms  # 不必要的开销
```
**修复**: 移除 `--render` 参数（headless 模式）

---

## 📚 参考文档

遇到问题时查看：

| 问题 | 查看文档 |
|------|---------|
| 相机慢 | `CAMERA_PERFORMANCE_OPTIMIZATION.md` |
| 物理慢 | `TERRAIN_OPTIMIZATION.md` |
| 性能分析 | `PERFORMANCE_PROFILING_QUICK_GUIDE.md` |
| 综合总结 | `PERFORMANCE_OPTIMIZATION_SUMMARY.md` |

---

## ✅ 检查完成！

所有检查项都通过后，您的配置已优化到最佳状态！

**预期性能**:
- 🚀 FPS: 45-60 steps/sec
- ⏱️ 单步耗时: 17-22ms
- 💾 GPU 显存: 1.5-2GB
- ⏰ 训练时间（3000 iter）: ~21 小时

**开始训练吧！** 🎉

```bash
python scripts/train.py --task=sirius_curriculum --num_envs=4096
```
