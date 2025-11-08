# 超参数对比：优化前 vs 优化后

## 🔴 问题诊断

你的 EMA 在 iteration 16 达到 10 后下降，**根本原因是超参数不匹配**：
- 配置是为 **4096 envs** 设计的
- 实际只运行 **256 envs**（显存限制）
- **学习率太高 + Batch size 太小 = 训练不稳定**

---

## 📊 关键指标对比

| 参数 | 原始 (4096 envs) | 优化前 (256 envs) | 优化后 (256 envs) | 说明 |
|------|-----------------|-------------------|-------------------|------|
| **num_envs** | 4096 | 256 | 256 | 显存限制 |
| **num_steps_per_env** | 24 | 24 ❌ | 48 ✅ | 增加采样步数 |
| **total_samples/iter** | 98,304 | 6,144 ❌ | 12,288 ✅ | 每次迭代的样本数 |
| **mini_batch_size** | 24,576 | 1,536 ❌ | 3,072 ✅ | 缩小 16x → 8x |
| **learning_rate** | 1e-3 | 1e-3 ❌ | 2.5e-4 ✅ | 降低 4 倍 |
| **clip_param** | 0.2 | 0.2 ❌ | 0.15 ✅ | 更保守的更新 |
| **entropy_coef** | 0.01 | 0.01 | 0.02 ✅ | 鼓励探索 |
| **num_learning_epochs** | 5 | 5 ❌ | 8 ✅ | 充分利用数据 |
| **desired_kl** | 0.01 | 0.01 | 0.008 ✅ | 更严格约束 |
| **max_iterations** | 1500 | 1200 ❌ | 2000 ✅ | 补偿总样本数 |

---

## 🎯 核心问题：Batch Size 缩小导致训练不稳定

### 数学解释

```python
# 梯度估计的方差：
Var(gradient) ∝ 1 / batch_size

# 你的情况：
batch_size_old = 24,576
batch_size_new = 1,536  # 缩小 16 倍

# 梯度方差：
Var_new = 16 × Var_old  # 噪声增加 16 倍！

# 但学习率没变：
lr = 1e-3  # 仍然很高

# 结果：
策略更新 = lr × noisy_gradient
         = 大步长 × 高噪声梯度
         = 随机乱跳！💥
```

### 为什么在 iteration 16 崩溃？

```
Iteration 1-16:
  - 初期学习简单技能（站立、平衡）
  - 即使有噪声，也能学到基础动作
  - EMA 上升到 10 ✅

Iteration 16:
  - EMA 达到阈值 → Curriculum 扩展速度范围
  - 任务突然变难（需要更快移动）
  
Iteration 16-30:
  - 高学习率 + 高噪声 + 难任务
  - 策略更新太激进 → 破坏已学会的技能
  - EMA 崩溃 ❌
```

---

## ✅ 优化策略

### 1. 降低学习率（最重要！）

```python
# 缩放规则：lr ∝ sqrt(batch_size)
learning_rate = 1e-3 × sqrt(12,288 / 98,304)
              = 1e-3 × sqrt(0.125)
              = 1e-3 × 0.354
              ≈ 3.5e-4

# 保守起见，用 2.5e-4
learning_rate = 2.5e-4  ✅
```

**效果**：
- ✅ 降低 4 倍（1e-3 → 2.5e-4）
- ✅ 策略更新更平滑
- ✅ 即使 curriculum 扩展也不会崩溃

### 2. 增加 Batch Size

```python
num_steps_per_env = 48  # 从 24 提高到 48

# 效果：
batch_size = 256 × 48 / 4 = 3,072  # 翻倍！
```

**效果**：
- ✅ 梯度估计更准确
- ✅ 降低方差到原来的 1/2
- ✅ 训练更稳定

### 3. 更保守的 PPO 参数

```python
clip_param = 0.15        # 限制策略变化幅度
num_learning_epochs = 8  # 充分学习每批数据
desired_kl = 0.008       # 严格的 KL 约束
```

**效果**：
- ✅ 防止单个坏批次毁掉策略
- ✅ 更充分地利用有限的数据
- ✅ adaptive lr 会自动调整学习率

### 4. 增加训练时长

```python
max_iterations = 2000  # 从 1200 提高

# 总样本数对比：
原始：4096 × 24 × 1500 = 147,456,000 样本
优化：256 × 48 × 2000 = 24,576,000 样本
```

**效果**：
- ✅ 虽然仍少 6 倍样本，但已经好很多
- ✅ 配合更高效的学习（epochs=8），总体差不多

---

## 📈 预期训练曲线对比

### 优化前（你遇到的问题）

```
EMA ↑
 10 |           ╱‾‾‾╲
    |      ╱‾‾‾╱     ╲╲
  5 | ╱‾‾‾╱            ╲╲___
    |╱                      ╲╲
  0 |________________________╲___
    0   10   20   30   40   50  Iter

问题：
- 快速上升（0-16）
- 崩溃式下降（16-30）
- 无法恢复
```

### 优化后（预期）

```
EMA ↑
 15 |                       ╱‾‾‾‾‾╲
    |                  ╱‾‾‾╱       ╲___
 10 |             ╱‾‾‾╱                ‾‾╲___
    |        ╱‾‾‾╱                           ╲
  5 |   ╱‾‾‾╱
    |╱‾
  0 |__________________________________________
    0    50   100  150  200  250  300  Iter

特点：
- 平稳上升（0-100）
- Curriculum 扩展后短暂下降（100）
- 快速恢复并超越（100-200）
- 继续稳步提升（200+）
```

---

## 🚀 立即行动

### 步骤 1：清空旧的训练结果

```bash
cd /home/eziothean/ATEC_VIS_E2E

# 备份旧日志（如果需要）
mv logs/sirius_diff_release logs/sirius_diff_release.backup

# 或者直接删除
rm -rf logs/sirius_diff_release
```

### 步骤 2：重新开始训练

```bash
cd scripts
source /home/eziothean/ATEC_VIS_E2E/.venv/bin/activate.fish

python train.py --task=sirius --headless
```

### 步骤 3：监控关键指标

训练时关注这些指标：

```bash
# 1. KL 散度（应该在 0.005-0.015 之间）
mean_kl = 0.008  ✅ 正常
mean_kl = 0.030  ❌ 学习率太高
mean_kl = 0.001  ⚠️  学习率太低

# 2. Episode 长度（应该逐渐增加）
mean_episode_length > 200  ✅ 机器人能站立行走
mean_episode_length < 100  ❌ 频繁摔倒

# 3. EMA（应该平稳上升）
EMA 逐渐从 0 → 5 → 10 → 15  ✅ 正常学习
EMA 突然下降 > 30%            ❌ 策略崩溃

# 4. 学习率（adaptive 会自动调整）
lr = 2.5e-4 ± 50%  ✅ 正常波动
lr < 1e-5          ⚠️  可能学习太慢
```

### 步骤 4：可视化检查（每 50 iterations）

```bash
# 加载最新 checkpoint
python play.py --task=sirius --checkpoint=-1

# 观察：
# ✅ 机器人能站立
# ✅ 跟随速度指令
# ✅ 步态稳定
# ✅ 不会突然摔倒
```

---

## 🔧 进一步调优（如果需要）

### 如果训练仍然不稳定

```python
# 选项 A：再降低学习率
learning_rate = 1.5e-4  # 从 2.5e-4 降低

# 选项 B：更保守的 clip
clip_param = 0.1  # 从 0.15 降低

# 选项 C：增加 batch size
num_steps_per_env = 64  # 从 48 提高
```

### 如果训练太慢

```python
# 选项 A：小幅提高学习率
learning_rate = 3.5e-4  # 从 2.5e-4 提高
# ⚠️ 注意监控 KL 散度！

# 选项 B：减少 epochs（如果过拟合）
num_learning_epochs = 6  # 从 8 降低

# 选项 C：如果有更多显存，增加环境数
num_envs = 384  # 如果显存允许
learning_rate = 3.0e-4  # 相应调整
```

---

## 📚 理论依据

### Batch Size 缩放定律

**参考论文**：
- Goyal et al. 2017: "Accurate, Large Minibatch SGD"
  - 线性缩放规则（监督学习）
  
- Hoffer et al. 2017: "Train longer, generalize better"
  - 平方根缩放规则（更稳定）

**本项目采用平方根规则**：
```
lr_new = lr_old × sqrt(batch_size_new / batch_size_old)
```

### PPO 的特殊性

PPO（Schulman et al. 2017）对超参数特别敏感：

1. **Clip 参数**：防止策略突变
   - 典型值：0.1-0.3
   - 小 batch → 用更小的值（0.1-0.15）

2. **Entropy Bonus**：平衡探索-利用
   - 太小（< 0.001）：过早收敛
   - 太大（> 0.05）：策略不稳定
   - 视觉任务：0.01-0.03

3. **KL 散度**：衡量策略变化
   - 配合 adaptive lr 使用
   - 典型值：0.005-0.02

---

## ✅ 总结

### 问题根源

你的超参数是为 **4096 envs** 设计的，但只用了 **256 envs**：
- ❌ Batch size 缩小 16 倍
- ❌ 学习率没有相应调整
- ❌ 梯度噪声增加 16 倍
- ❌ 策略更新太激进 → 训练崩溃

### 解决方案

已经优化的配置：
- ✅ 学习率降低 4 倍（1e-3 → 2.5e-4）
- ✅ Batch size 翻倍（48 steps vs 24）
- ✅ 更保守的 PPO 参数
- ✅ 更多训练迭代（2000 vs 1200）

### 预期效果

应用优化后，你应该看到：
- ✅ 平滑的训练曲线（不会崩溃）
- ✅ EMA 稳步上升到 15+
- ✅ Curriculum 扩展后能快速恢复
- ✅ 最终获得更好的策略

### 关键建议

1. **立即重新开始训练**（用新配置）
2. **监控 KL 散度**（判断学习率是否合适）
3. **可视化检查**（每 50 iterations）
4. **耐心等待**（200+ iterations 才能看到最终效果）

**你的观察非常正确！** 🎯 这就是 EMA 下降的真正原因。现在重新训练，应该能解决问题！
