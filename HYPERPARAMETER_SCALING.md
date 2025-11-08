# 超参数缩放指南：从 4096 envs 到 256 envs

## 🎯 问题根源

原始超参数是为 **4096 个并行环境** 设计的，但由于显存限制（每个相机 40MB），你只能运行 **256 个环境**。

**核心问题**：Batch size 缩小了 **16 倍**，但学习率没有相应调整！

## 📊 Batch Size 对比

### 原始配置（4096 envs）

```python
num_envs = 4096
num_steps_per_env = 24
num_mini_batches = 4

total_samples = 4096 × 24 = 98,304 样本/iteration
mini_batch_size = 98,304 / 4 = 24,576 样本/mini_batch
```

### 你的配置（256 envs，未调整）

```python
num_envs = 256
num_steps_per_env = 24        # 未改变
num_mini_batches = 4          # 未改变

total_samples = 256 × 24 = 6,144 样本/iteration  ❌ 缩小 16 倍
mini_batch_size = 6,144 / 4 = 1,536 样本/mini_batch  ❌ 缩小 16 倍
```

### 优化后的配置（256 envs，已调整）

```python
num_envs = 256
num_steps_per_env = 48        # ✅ 提高到 48（翻倍）
num_mini_batches = 4          # 保持不变

total_samples = 256 × 48 = 12,288 样本/iteration  ✅ 翻倍
mini_batch_size = 12,288 / 4 = 3,072 样本/mini_batch  ✅ 翻倍
```

## 🔧 超参数调整规则

### 1. 学习率（Learning Rate）

**规则**：学习率应与 `sqrt(batch_size)` 成正比

```python
# 原始：4096 envs
lr_original = 1e-3
batch_size_original = 4096 × 24 = 98,304

# 目标：256 envs
batch_size_new = 256 × 48 = 12,288

# 缩放因子
scale = sqrt(batch_size_new / batch_size_original)
      = sqrt(12,288 / 98,304)
      = sqrt(0.125)
      ≈ 0.354

# 新学习率
lr_new = lr_original × scale
       = 1e-3 × 0.354
       ≈ 3.5e-4

# 保守起见，使用 2.5e-4（稍微低一点更稳定）
learning_rate = 2.5e-4  ✅
```

**为什么这样做？**
- **小 batch size** → 梯度估计噪声更大 → 需要更小的学习率
- **大 batch size** → 梯度估计更准确 → 可以用更大的学习率

### 2. 增加每次迭代的步数（Steps per Env）

```python
# 目标：增大 batch size 以补偿环境数减少
num_steps_per_env = 48  # 从 24 提高到 48

# 效果：
# - batch_size 从 6,144 提高到 12,288
# - 每次策略更新使用更多数据
# - 梯度估计更稳定
```

### 3. 增加学习轮数（Learning Epochs）

```python
# 原始：5 epochs（每次迭代对数据学习 5 遍）
# 优化：8 epochs

num_learning_epochs = 8  # 从 5 提高到 8

# 原因：
# - 小 batch size 意味着每个 epoch 看到的数据少
# - 增加 epochs 可以更充分地利用数据
# - 但不要太高（>10），否则会过拟合
```

### 4. 降低 PPO Clip 范围

```python
clip_param = 0.15  # 从 0.2 降低到 0.15

# 原因：
# - 小 batch size → 价值估计方差大
# - 限制策略更新幅度 → 更保守 → 更稳定
# - 防止单个坏批次导致性能崩溃
```

### 5. 提高 Entropy Bonus

```python
entropy_coef = 0.02  # 从 0.01 提高到 0.02

# 原因：
# - 鼓励探索（action 分布更宽）
# - 防止过早收敛到次优策略
# - 特别对视觉输入重要（需要探索不同视角）
```

### 6. 降低 KL 散度约束

```python
desired_kl = 0.008  # 从 0.01 降低到 0.008

# 原因：
# - adaptive learning rate 会根据 KL 散度调整 lr
# - 更严格的 KL 约束 → lr 自动降低 → 更稳定
# - 防止策略突变
```

### 7. 增加总训练迭代数

```python
max_iterations = 2000  # 从 1200 提高到 2000

# 原因：
# - 每次迭代的样本数少了（256 vs 4096）
# - 需要更多迭代才能看到相同数量的样本
# - 总样本数对比：
#   原始：4096 × 24 × 1200 = 117,964,800 样本
#   优化：256 × 48 × 2000 = 24,576,000 样本（仍少 5 倍）
```

## 📈 预期效果对比

### 原始配置（4096 envs）

```
优点：
✅ 大 batch size → 梯度估计准确
✅ 训练快速收敛
✅ 稳定性高

缺点：
❌ 显存需求 160 GB+（相机）
❌ 无法在单卡上运行
```

### 未调整配置（256 envs + 原始超参）

```
问题：
❌ 小 batch size (6,144) + 高学习率 (1e-3)
❌ 梯度噪声大 → 训练震荡
❌ EMA 上升后快速下降（你遇到的问题）
❌ 策略不稳定，容易崩溃

典型现象：
- Iteration 1-16:  快速学习 → EMA 达到 10
- Iteration 16:    梯度噪声 + 高 lr → 策略突变
- Iteration 16-30: 性能崩溃 → EMA 下降
```

### 优化后配置（256 envs + 调整后超参）

```
改进：
✅ 中等 batch size (12,288) + 适配学习率 (2.5e-4)
✅ 更保守的更新（clip=0.15）
✅ 更充分的数据利用（epochs=8）
✅ 更多探索（entropy=0.02）

预期训练曲线：
Iteration 0-50:   平稳上升，EMA 0 → 5
Iteration 50-100: 继续上升，EMA 5 → 8
Iteration 100:    Curriculum 扩展，EMA 暂时下降
Iteration 100-200:适应新难度，EMA 恢复
Iteration 200+:   稳定提升，EMA 10 → 15+
```

## 🎯 调优策略

### 如果训练仍然不稳定

#### 1. 进一步降低学习率

```python
learning_rate = 1.5e-4  # 从 2.5e-4 再降低
```

#### 2. 增加 batch size

```python
num_steps_per_env = 64  # 从 48 提高到 64
# batch_size = 256 × 64 / 4 = 4,096
```

#### 3. 更保守的 PPO 参数

```python
clip_param = 0.1        # 更小的更新幅度
num_learning_epochs = 5 # 减少过拟合风险
```

### 如果训练太慢

#### 1. 提高学习率（小心！）

```python
learning_rate = 3.5e-4  # 从 2.5e-4 提高
# 但需要监控 KL 散度，防止策略突变
```

#### 2. 减少 epochs

```python
num_learning_epochs = 6  # 从 8 降低到 6
```

### 如果显存允许，增加环境数

```python
# 每增加 256 envs → 额外 10 GB 显存
num_envs = 512  # 如果你有 24GB+ 显存
max_envs = 512

# 相应调整学习率：
learning_rate = 2.5e-4 × sqrt(512/256) = 2.5e-4 × 1.414 ≈ 3.5e-4
```

## 📋 完整的优化配置

### sirius_flat_config.py

```python
class SiriusFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 256  # 显存限制

class SiriusFlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # 核心超参数（针对 256 envs 优化）
        learning_rate = 2.5e-4      # ✅ 降低学习率
        clip_param = 0.15           # ✅ 更保守的更新
        entropy_coef = 0.02         # ✅ 鼓励探索
        num_learning_epochs = 8     # ✅ 更充分学习
        desired_kl = 0.008          # ✅ 严格的 KL 约束
        
        # 保持不变的参数
        num_mini_batches = 4        # ✅ 保持
        value_loss_coef = 1.0       # ✅ 保持
        gamma = 0.99                # ✅ 保持
        lam = 0.95                  # ✅ 保持
        max_grad_norm = 1.0         # ✅ 保持

    class runner( LeggedRobotCfgPPO.runner ):
        num_steps_per_env = 48      # ✅ 增大 batch size
        max_iterations = 2000       # ✅ 更多迭代
```

## 🔍 监控指标

训练时重点关注：

### 1. KL 散度（KL Divergence）

```bash
# 在训练日志中查找：
mean_policy_loss | mean_value_loss | mean_kl

# 正常范围：
KL = 0.005 - 0.015  ✅ 正常
KL > 0.05           ❌ 策略更新太激进，降低 lr
KL < 0.001          ⚠️  学习太慢，可能提高 lr
```

### 2. 学习率自适应

```bash
# adaptive schedule 会根据 KL 调整 lr：
if KL > 1.5 × desired_kl:  lr ← lr × 0.5  (减半)
if KL < 0.5 × desired_kl:  lr ← lr × 1.5  (增加)
```

### 3. Episode 长度

```bash
# 如果机器人频繁摔倒：
mean_episode_length < 100  ❌ 策略不稳定
mean_episode_length > 200  ✅ 正常学习
```

### 4. Reward 分解

```bash
# 不仅看 EMA，还要看：
tracking_lin_vel:  速度追踪奖励（你的 EMA）
orientation:       姿态奖励（是否直立）
base_height:       高度奖励（是否站立）
feet_air_time:     步态奖励（是否正常行走）
```

## 📊 理论依据

### Batch Size 缩放定律

**线性缩放规则**（Goyal et al., 2017）：
```
当 batch_size 增大 k 倍时，learning_rate 也应增大 k 倍
```

**平方根缩放规则**（Hoffer et al., 2017）：
```
当 batch_size 增大 k 倍时，learning_rate 应增大 sqrt(k) 倍
```

**本项目采用平方根规则**，因为：
- 更保守，训练更稳定
- 适合 RL（比监督学习噪声更大）
- 经验证明对 PPO 效果更好

### PPO 的特殊考虑

PPO（Proximal Policy Optimization）对超参数敏感：

1. **Clip 范围**：限制策略更新幅度
   - 小 batch size → 价值估计方差大 → 需要更小的 clip
   
2. **Entropy Bonus**：鼓励探索
   - RL 需要探索 → 不要太小（< 0.001）
   - 视觉任务 → 需要更多探索（0.01-0.03）

3. **KL 散度**：衡量策略变化程度
   - 配合 adaptive lr 使用
   - 防止"cliff"（性能悬崖式下降）

## ⚠️ 常见错误

### 错误 1：只减少环境数，不调整超参

```python
❌ num_envs = 256  # 从 4096 减少
❌ learning_rate = 1e-3  # 没有调整！

结果：训练震荡，性能崩溃
```

### 错误 2：线性缩放学习率

```python
❌ lr_new = 1e-3 × (256/4096) = 6.25e-5  # 太小了！

结果：训练极慢，可能陷入局部最优
```

### 错误 3：忽略总样本数

```python
❌ max_iterations = 1200  # 保持不变

实际总样本：256 × 24 × 1200 = 7,372,800
原始总样本：4096 × 24 × 1200 = 117,964,800

差距：16 倍！训练不充分
```

### 错误 4：过度调整所有参数

```python
❌ 同时改变 10+ 个超参数

结果：无法判断哪个改变有效，调参变成碰运气
```

**正确做法**：
1. 先调整 learning_rate 和 num_steps_per_env（最重要）
2. 观察训练 50-100 iterations
3. 根据表现微调其他参数

## 🎓 总结

### 核心原则

1. **Batch Size 决定一切**
   - 256 envs → batch_size 缩小 16 倍
   - 学习率必须相应调整（sqrt 规则）

2. **稳定性优先**
   - 宁可训练慢一点，也不要不稳定
   - 降低 lr、clip、KL，增加稳定性

3. **充分利用数据**
   - 增加 num_steps_per_env（更大 batch）
   - 增加 num_learning_epochs（多学几遍）

4. **监控指标**
   - KL 散度：策略更新是否合理
   - Episode 长度：策略是否崩溃
   - Reward 分解：具体哪里出问题

### 预期改进

应用这些超参数后，你应该看到：

✅ **更平滑的训练曲线**
- EMA 稳步上升，波动更小
- 即使 curriculum 扩展，也能快速恢复

✅ **更高的最终性能**
- EMA 可以达到 15+（vs 之前的 8-10）
- 机器人行走更稳定、更快速

✅ **更可预测的训练过程**
- 不会出现"突然崩溃"的现象
- Curriculum 扩展按预期工作

### 如果仍有问题

检查这些方面：
1. CNN encoder 是否正确初始化？
2. 视觉观测是否正确归一化？
3. Curriculum 参数是否太激进？
4. 是否有其他 bug（duplicate step 等）？

记住：**超参数调优是科学 + 艺术**，需要耐心和实验！
