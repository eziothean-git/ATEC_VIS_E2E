# GPU 利用率优化指南

## 问题分析

### 观察到的现象
- **模拟阶段**: GPU 利用率 ~60% (低于预期)
- **学习阶段**: GPU 利用率 ~100% (正常)
- **整体**: 不连续的利用率，平均低于原始的 90%

### 根本原因

#### 1. 环境数减少 (4096 → 1024)
```
并行度降低 → GPU 内核利用不足 → 利用率下降
```

**影响**:
- 物理模拟的并行度降低 75%
- 相机渲染的批量大小减少 75%
- GPU 上的线程束 (warp) 利用率降低
- 数据传输开销相对增加

#### 2. 模拟和学习阶段不平衡
```
模拟 (60%) ←→ 学习 (100%)
      ↓
   平均利用率降低
```

**原因**:
- 小批量模拟时，GPU 部分空闲
- 学习阶段计算密集，GPU 饱和
- 两阶段切换时有开销

#### 3. 视觉模块增加的复杂度
```
深度图获取 → CPU/GPU 数据传输 → 可能的瓶颈
```

## 优化方案

### 方案 1: 调整 PPO 超参数 ✅ (已应用)

**目标**: 增加学习阶段的计算量，提高 GPU 利用率

**修改**:
```python
class algorithm:
    learning_rate = 2.5e-4  # 降低学习率
    num_learning_epochs = 8  # 5 → 8 (增加 60%)
    num_mini_batches = 2     # 4 → 2 (增大批量)
```

**效果**:
- ✅ 学习阶段时间增加 → GPU 利用率更平衡
- ✅ 更大的 mini-batch (12288 vs 6144) → 更好的 GPU 利用
- ✅ 更多 epochs → 每次采样的学习效率提高
- ⚠️ 可能增加过拟合风险 (需监控)

**计算量对比**:
```
原始: 5 epochs × 4 batches × 24576 samples = 491,520 梯度计算
优化: 8 epochs × 2 batches × 12288 samples = 196,608 梯度计算
比例: 40% (仍少于原始，但 GPU 利用率更高)
```

### 方案 2: 增加环境数 (如果内存允许)

**选项 A: 适度增加**
```python
num_envs = 1536  # 1024 → 1536 (+50%)
```
- GPU 利用率提升到 ~75-80%
- 显存需求增加 ~50%
- 学习率可调整为 3e-4

**选项 B: 尽可能增加**
```python
num_envs = 2048  # 1024 → 2048 (+100%)
```
- GPU 利用率提升到 ~80-85%
- 显存需求增加 ~100%
- 学习率可调整为 5e-4

### 方案 3: 优化数据流 (高级)

#### A. 重叠计算和传输
```python
# 在 OnPolicyRunner.learn() 中使用 CUDA streams
# 让 CPU-GPU 传输与 GPU 计算重叠
```

#### B. 预取深度图像
```python
# 在环境步进时提前启动下一帧的相机渲染
# 减少 GPU 等待时间
```

#### C. 异步环境步进
```python
# 使用 Isaac Gym 的异步 API
# 让模拟和学习部分重叠
```

### 方案 4: 调整 num_steps_per_env

**当前**: 24 步 (默认)

**选项**: 增加到 48
```python
class runner:
    num_steps_per_env = 48  # 24 → 48
```

**效果**:
- ✅ 模拟阶段时间翻倍 → 更好的 GPU 利用
- ✅ 减少学习频率 → 减少切换开销
- ⚠️ 增加 on-policy 数据的 "陈旧度"
- ⚠️ 可能影响训练稳定性

### 方案 5: 相机优化

#### A. 减少相机数量
```python
class camera:
    max_envs = 512  # 只为一半环境创建相机
```
- 减少渲染开销
- 仍能获得足够的视觉信息

#### B. 降低相机分辨率
```python
class camera:
    width = 65   # 87 → 65 (-25%)
    height = 43  # 58 → 43 (-26%)
```
- 减少显存和带宽
- CNN 输入更小 → 计算更快
- 可能轻微降低视觉信息质量

#### C. 使用更高效的相机模式
```python
class camera:
    use_collision_geometry = False  # 更快的渲染
    enable_tensors = True           # GPU tensor 路径
```

## 推荐的优化组合

### 保守方案 (已应用)
```python
num_envs = 1024
num_learning_epochs = 8
num_mini_batches = 2
learning_rate = 2.5e-4
```
**预期**: GPU 利用率提升到 ~70-75%

### 激进方案 (需要测试内存)
```python
num_envs = 1536
num_learning_epochs = 8
num_mini_batches = 2
num_steps_per_env = 48
learning_rate = 3e-4
camera.max_envs = 768  # 只为一半环境创建相机
```
**预期**: GPU 利用率提升到 ~80-85%

### 极限方案 (如果显存充足)
```python
num_envs = 2048
num_learning_epochs = 5  # 恢复默认
num_mini_batches = 4     # 恢复默认
learning_rate = 5e-4
```
**预期**: GPU 利用率接近原始的 ~90%

## 监控指标

### 训练过程中监控
```bash
# 1. GPU 利用率
watch -n 1 nvidia-smi

# 2. TensorBoard 指标
tensorboard --logdir logs/sirius_flat
# 关注:
#   - Perf/total_fps (越高越好)
#   - Perf/collection time (模拟时间)
#   - Perf/learn time (学习时间)
#   - Loss/* (确保不过拟合)

# 3. 显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

### 性能指标目标
- **FPS**: >5000 (1024 envs)
- **GPU 利用率**: >75%
- **显存使用**: <10GB (RTX 3090)
- **训练稳定性**: KL divergence < 0.05

## 实验建议

### 第一步: 测试当前优化
```bash
python train.py --task=sirius --num_envs=1024 --headless --max_iterations=100
```
观察:
- GPU 利用率是否提升
- FPS 是否合理
- 训练是否稳定

### 第二步: 尝试增加环境数
```bash
# 测试 1536 envs
python train.py --task=sirius --num_envs=1536 --headless --max_iterations=100
```
如果显存溢出，回退到 1024

### 第三步: 调整 steps_per_env
```bash
# 修改配置后测试 48 steps
python train.py --task=sirius --num_envs=1024 --headless --max_iterations=100
```
观察训练曲线是否更平滑

## 权衡考虑

### GPU 利用率 vs 训练质量
- 高 GPU 利用率 ≠ 更快收敛
- 可能需要在利用率和样本效率间平衡
- **原则**: 优先保证训练稳定性

### 显存限制
- 视觉 RL 比纯 proprioception 显存需求更高
- 深度图 buffer: `24 × num_envs × 1 × 58 × 87 × 4 bytes`
  - 1024 envs: ~11.5 MB
  - 2048 envs: ~23 MB
  - 4096 envs: ~46 MB

### 训练时间
```
原始 (4096 envs, 5 epochs): ~X 小时
优化 (1024 envs, 8 epochs): ~1.6X 小时 (预估)
激进 (1536 envs, 8 epochs): ~1.2X 小时 (预估)
```

## 下一步行动

1. ✅ 应用保守优化 (已完成)
2. ⏳ 运行训练观察 GPU 利用率
3. ⏳ 根据显存情况考虑增加 num_envs
4. ⏳ 监控训练曲线确保稳定性
5. ⏳ 必要时回退或进一步调整

## 参考

- Isaac Gym Performance Guide
- PPO Hyperparameter Tuning Best Practices
- Vision-based RL Optimization Techniques
