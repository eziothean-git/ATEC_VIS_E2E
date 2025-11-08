# EMA 指标详解：为什么达到 10 后又下降？

## 📊 什么是 EMA？

### 定义
**EMA = Exponential Moving Average（指数移动平均）**

在你的训练中，EMA 具体指 **`command_tracking_ema`**，用于衡量机器人追踪速度指令的能力。

### 计算公式

```python
# 每次环境重置时更新
vals = episode_sums["tracking_lin_vel"] / max_episode_length_s
mean_val = np.mean(vals)  # 所有环境的平均追踪奖励

# EMA 更新（alpha = 0.2）
ema = 0.8 * ema_old + 0.2 * mean_val

# 或者写成：
ema_new = (1 - α) * ema_old + α * current_value
# α = 0.2 表示新值占 20%，旧值占 80%
```

### 具体含义

```python
# tracking_lin_vel 奖励的计算：
lin_vel_error = ||commanded_velocity - actual_velocity||²
tracking_reward = exp(-lin_vel_error / sigma)

# 含义：
# - 机器人实际速度接近指令速度 → tracking_reward 接近 1.0
# - 机器人速度偏离指令 → tracking_reward 下降（指数衰减）
# - 机器人完全不动或乱跑 → tracking_reward 接近 0

# episode_sums["tracking_lin_vel"] = Σ tracking_reward 整个 episode
# 然后除以 episode 时长（秒），得到平均每秒的追踪奖励
```

### EMA 值的含义

| EMA 值 | 含义 | 机器人表现 |
|--------|------|-----------|
| 0-2    | 很差 | 无法站立或随机移动 |
| 2-5    | 较差 | 能站立，但不跟随指令 |
| 5-8    | 中等 | 开始跟随指令，但不稳定 |
| 8-10   | 良好 | 较好地跟随速度指令 |
| 10-15  | 优秀 | 精确跟随指令 |
| 15+    | 极好 | 接近完美追踪 |

## 🔍 为什么 16 轮达到 10 后又下降？

### 现象分析

```
Iteration 1-16:  EMA 快速上升 0 → 10 ✅
Iteration 16-30: EMA 下降 10 → ? ❌
```

这是**训练不稳定的典型症状**，可能的原因：

### 原因 1：Curriculum 扩展导致难度突增 ⚠️

```python
# Curriculum 机制：
# 当 EMA 达到阈值（0.8）并连续成功多次时，扩展速度指令范围

# 假设：
Iteration 1-10:  command_range = [-0.5, 0.5] m/s  → 机器人学会了
Iteration 10:    EMA = 10 (超过阈值) → 扩展！
Iteration 11:    command_range = [-0.7, 0.7] m/s  → 突然变难
Iteration 11-20: 机器人还不会快速移动 → EMA 下降

# 这是正常的 curriculum 过程！
```

**检查方法**：
```bash
# 查看训练日志，是否有类似输出：
[Curriculum] ✓ EXPANDED COMMAND RANGE
  lin_vel_x: 0.500 → 0.700 m/s
```

如果有，说明 curriculum 在工作，EMA 下降是**预期的暂时性下降**。

### 原因 2：PPO 的探索-利用权衡 🎲

```python
# PPO 在训练过程中需要平衡：
# - Exploitation（利用）：使用当前最优策略
# - Exploration（探索）：尝试新动作发现更好策略

# 可能的情况：
Iteration 1-16:  主要 exploitation → 学会基本技能 → EMA 上升
Iteration 16:    策略开始探索新动作 → 短期性能下降
Iteration 16-30: 探索新的运动模式 → EMA 波动
Iteration 30+:   找到更好的策略 → EMA 恢复并超越之前
```

**特征**：
- EMA 下降幅度不大（10 → 8）
- 训练曲线有波动，但整体趋势向上
- 后续会恢复并超越之前的峰值

### 原因 3：过拟合到初始条件 ⚠️

```python
# 如果机器人只学会了特定情况下的行走：
# - 特定的初始姿态
# - 特定的速度指令
# - 没有泛化能力

# 当遇到新情况时：
# - Domain randomization 增加
# - 速度指令范围扩大
# - 地形变化
# → 策略失效，EMA 下降
```

**检查方法**：
```python
# 在 sirius_flat_config.py 中查看：
class domain_rand:
    randomize_base_mass = True
    added_mass_range = [-5., 5.]
    friction_range = [0., 1.5]

# 如果这些参数在训练中增加，可能导致性能下降
```

### 原因 4：学习率衰减导致的震荡 📉

```python
# PPO 的学习率通常会衰减：
# lr = lr_initial × (1 - iteration / max_iterations)

# 可能的情况：
Iteration 1-16:  高学习率 → 快速学习 → EMA 上升
Iteration 16:    学习率衰减 → 策略更新幅度变小
Iteration 16-30: 陷入局部最优 → EMA 停滞或下降
```

### 原因 5：视觉特征过拟合 👁️

```python
# 特别是你刚修复了相机同步问题，可能出现：

# 情况 A：视觉噪声敏感
# - 前期主要用 proprio（关节角度、IMU）
# - 中期开始依赖视觉
# - 视觉信息有噪声或不稳定 → 策略混乱

# 情况 B：视觉-运动耦合不稳定
# - 机器人学会了视觉→动作的映射
# - 但这个映射在新情况下泛化性差
# - 导致性能下降
```

## 🔧 诊断方法

### 1. 查看完整的训练曲线

```bash
# 使用 TensorBoard 或查看日志
tensorboard --logdir logs/

# 关注指标：
# - mean_reward: 总体奖励趋势
# - tracking_lin_vel: 速度追踪奖励
# - episode_length: episode 长度（是否提前终止）
```

**正常情况**：
```
EMA:   0 → 10 → 8 → 12 → 15 (先升后降再升，整体向上)
```

**异常情况**：
```
EMA:   0 → 10 → 5 → 3 → 2 (持续下降，策略崩溃)
```

### 2. 检查 Curriculum 日志

```bash
# 查看训练输出，搜索：
grep "EXPANDED COMMAND RANGE" logs/latest_log.txt

# 如果在 iteration 16 附近有扩展，说明难度增加是正常的
```

### 3. 查看其他奖励项

```python
# EMA 只反映速度追踪，还需要看其他奖励：
# - orientation: 身体姿态是否稳定
# - base_height: 是否保持站立高度
# - feet_air_time: 步态是否正常
# - dof_vel: 关节速度是否合理

# 如果其他奖励也下降 → 策略整体崩溃
# 如果只有 tracking 下降 → 可能是 curriculum 难度增加
```

### 4. 可视化机器人行为

```bash
# 加载 checkpoint 并可视化
python legged_gym/scripts/play.py \
  --task=sirius \
  --load_run=logs/sirius_diff_release/YYYY-MM-DD_HH-MM-SS

# 观察：
# - 机器人是否还能站立？
# - 是否跟随速度指令？
# - 步态是否正常？
```

### 5. 对比不同 iteration 的表现

```bash
# 加载 iteration 16 的模型
python play.py --checkpoint=16  # EMA = 10

# 加载 iteration 30 的模型
python play.py --checkpoint=30  # EMA 下降后

# 对比：
# - 16 轮的模型是否更稳定？
# - 30 轮的模型是否在尝试更快的速度？
```

## ✅ 可能的解决方案

### 方案 1：调整 Curriculum 参数（推荐）

```python
# sirius_flat_config.py
class commands:
    # 降低扩展速度
    curriculum_increment = 0.01  # 从 0.02 降低
    
    # 提高扩展阈值
    curriculum_progress_threshold = 0.9  # 从 0.8 提高
    
    # 增加连续成功要求
    curriculum_consecutive_successes = 10  # 从 5 提高
    
    # 增加扩展间隔
    curriculum_min_resets_between_expansions = 50  # 从 20 提高
```

**效果**：curriculum 扩展更谨慎，避免难度突增

### 方案 2：降低学习率（如果是震荡）

```python
# sirius_flat_config.py
class runner(LeggedRobotCfgPPO.runner):
    policy_learning_rate = 1e-4  # 从 1e-3 降低
    value_learning_rate = 1e-4
```

### 方案 3：增加训练稳定性

```python
class algorithm(LeggedRobotCfgPPO.algorithm):
    # 减小 PPO clip range（更保守的更新）
    clip_param = 0.1  # 从 0.2 降低
    
    # 增加 entropy bonus（鼓励探索）
    entropy_coef = 0.01
    
    # 使用 GAE 平滑优势估计
    use_gae = True
    gae_lambda = 0.95
```

### 方案 4：增加训练时长

```python
class runner:
    max_iterations = 2000  # 从 1200 增加
    
    # 更长的 episode
    max_episode_length = 500  # 从 300 增加
```

**原因**：给策略更多时间适应新的难度

### 方案 5：调整奖励权重

```python
class rewards:
    class scales:
        # 降低 tracking 权重，增加稳定性奖励
        tracking_lin_vel = 0.8  # 从 1.0 降低
        orientation = -10.0     # 从 -5.0 加强
        base_height = -300.0    # 从 -200 加强
```

**效果**：优先保证稳定性，再追求速度

## 📈 正常的训练曲线应该是什么样？

### 典型的 RL 训练曲线

```
EMA ↑
 15 |                        ╱‾‾‾‾‾╲
    |                   ╱‾‾‾╱       ╲___
 10 |              ╱‾‾‾╱                ‾‾╲___
    |         ╱‾‾‾╱                           ╲
  5 |    ╱‾‾‾╱
    | ╱‾‾
  0 |___________________________________________
    0    16    32    48    64    80    96   Iter

阶段解释：
0-16:   快速学习基础技能（站立、平衡）
16-32:  Curriculum 扩展，难度增加，EMA 暂时下降
32-48:  适应新难度，EMA 恢复
48-64:  再次扩展，再次下降
64+:    稳定提升，逐渐逼近最优
```

**关键特征**：
- ✅ 有波动是正常的（尤其是 curriculum）
- ✅ 每次下降后应该能恢复
- ✅ 整体趋势向上
- ❌ 如果持续下降不恢复 → 有问题

### 你的情况分析

```
EMA ↑
 10 |           ╱‾‾‾╲
    |      ╱‾‾‾╱     ╲
  5 | ╱‾‾‾╱            ╲__ ← 你在这里
    |╱                     ╲
  0 |_______________________
    0   16   30          Iter
```

**判断**：
- 如果 iteration 30-50 能恢复 → **正常**，是 curriculum 效果
- 如果持续下降到 < 5 → **有问题**，需要调整

## 🎯 建议操作

### 立即行动：

1. **继续训练到 iteration 50-100**
   - 观察 EMA 是否回升
   - 如果回升 → 正常的 curriculum 过程
   - 如果持续下降 → 需要调整

2. **查看训练日志**
   ```bash
   tail -f logs/sirius_diff_release/*/summaries.txt
   # 找到 iteration 16 附近是否有 curriculum 扩展
   ```

3. **可视化当前模型**
   ```bash
   python play.py --checkpoint=latest
   # 看机器人是否还能正常行走
   ```

### 如果确认有问题：

1. **回退到 iteration 16 的 checkpoint**
2. **调整 curriculum 参数**（更保守）
3. **重新训练**

### 如果是正常的 curriculum：

1. **继续训练，不要停**
2. **预期 EMA 会在 30-50 轮后恢复**
3. **最终应该超过之前的 10**

## 总结

**EMA (command_tracking_ema)**：
- 指数移动平均的速度追踪奖励
- 衡量机器人跟随速度指令的能力
- 值越高越好（0-15 正常范围）

**16 轮达到 10 后下降**：
- 很可能是 curriculum 扩展导致难度增加
- 这是**正常现象**，不是 bug
- 需要继续训练观察是否恢复

**下一步**：
1. 继续训练到 50-100 轮
2. 观察 EMA 是否回升
3. 如果持续下降，调整 curriculum 参数

**关键指标不仅仅是 EMA**：
- 还要看总奖励、episode 长度、其他奖励项
- 最重要的是可视化机器人的实际表现！
