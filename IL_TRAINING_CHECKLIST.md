# Sirius IL训练实施检查清单

## 📋 准备阶段

### 1. 教师模型验证
- [ ] 确认已有训练好的 sirius 模型 (task=sirius)
- [ ] 记录模型路径: `logs/sirius/____/model___.pt`
- [ ] 验证教师模型性能 (RL reward > 80)
- [ ] 测试教师模型能否正常加载

```bash
# 测试教师模型
python legged_gym/scripts/play.py \
    --task=sirius \
    --load_run=logs/sirius/YYYY-MM-DD/HH-MM-SS
```

### 2. 代码修改

#### 2.1 修改 PPO 算法
- [ ] 打开 `rsl_rl/rsl_rl/algorithms/ppo.py`
- [ ] 在 `__init__` 中添加: `self.teacher_actor_critic = None`
- [ ] 在 `update()` 方法中添加IL损失计算逻辑
- [ ] 修改 `update()` 返回值包含 `mean_imitation_loss`

<details>
<summary>📝 点击查看代码片段</summary>

```python
# 在 PPO.update() 中添加 (位置: 计算actor_loss之后)

if (hasattr(self, 'teacher_actor_critic') and 
    self.teacher_actor_critic is not None and
    hasattr(self.cfg, 'use_imitation_loss') and 
    self.cfg.use_imitation_loss):
    
    with torch.no_grad():
        teacher_actions = self.teacher_actor_critic.act_inference(obs_batch)
    
    student_actions = self.actor_critic.act_inference(obs_batch)
    imitation_loss = F.mse_loss(student_actions, teacher_actions)
    
    il_coef = self.cfg.imitation_loss_coef
    actor_loss = actor_loss + il_coef * imitation_loss
    mean_imitation_loss = imitation_loss.item()
else:
    mean_imitation_loss = 0.0

# 修改返回值
return mean_value_loss, mean_surrogate_loss, mean_imitation_loss
```
</details>

#### 2.2 修改 OnPolicyRunner
- [ ] 打开 `rsl_rl/rsl_rl/runners/on_policy_runner.py`
- [ ] 在 `__init__` 末尾添加教师模型加载逻辑
- [ ] 添加 `load_teacher_model()` 方法
- [ ] 在 `learn()` 中记录IL损失到tensorboard

<details>
<summary>📝 点击查看代码片段</summary>

```python
# 在 OnPolicyRunner.__init__ 末尾添加
if hasattr(train_cfg.runner, 'teacher_model_path') and train_cfg.runner.teacher_model_path:
    self.load_teacher_model(train_cfg.runner.teacher_model_path)

# 添加新方法
def load_teacher_model(self, model_path):
    from rsl_rl.modules import ActorCritic
    
    print(f"[Teacher Model] Loading from: {model_path}")
    
    teacher_cfg = {
        'init_noise_std': 1.0,
        'actor_hidden_dims': [256, 128, 64],
        'critic_hidden_dims': [256, 128, 64],
        'activation': 'elu'
    }
    
    self.teacher_actor_critic = ActorCritic(
        self.env.cfg.env.num_observations,
        self.env.cfg.env.num_observations,
        self.env.cfg.env.num_actions,
        **teacher_cfg
    ).to(self.device)
    
    checkpoint = torch.load(model_path, map_location=self.device)
    teacher_state = checkpoint.get('model_state_dict', checkpoint.get('actor_critic', checkpoint))
    self.teacher_actor_critic.load_state_dict(teacher_state)
    
    for param in self.teacher_actor_critic.parameters():
        param.requires_grad = False
    self.teacher_actor_critic.eval()
    
    self.alg.teacher_actor_critic = self.teacher_actor_critic
    print("✅ Teacher model loaded and frozen")
```
</details>

#### 2.3 修改训练脚本
- [ ] 打开 `legged_gym/scripts/train.py`
- [ ] 添加 `--teacher_model` 命令行参数
- [ ] 将参数传递到 `train_cfg.runner.teacher_model_path`

<details>
<summary>📝 点击查看代码片段</summary>

```python
# 在参数解析部分添加
parser.add_argument('--teacher_model', type=str, default=None, 
                   help='Path to teacher model for IL training')
args = parser.parse_args()

# 在创建runner之前添加
if args.teacher_model:
    train_cfg.runner.teacher_model_path = args.teacher_model
    print(f"[IL Training] Using teacher model: {args.teacher_model}")
```
</details>

### 3. 配置文件检查
- [x] `sirius_curriculum_il_config.py` 已创建 ✅
- [x] `sirius_curriculum_finetune_config.py` 已创建 ✅
- [x] 任务已注册到 `__init__.py` ✅

---

## 🚀 阶段1: 平地IL训练

### 1. 启动训练
- [ ] 设置教师模型路径变量

```bash
export TEACHER_MODEL="logs/sirius/YYYY-MM-DD/HH-MM-SS/model_1800.pt"
```

- [ ] 运行阶段1训练

```bash
python legged_gym/scripts/train.py \
    --task=sirius_curriculum_il \
    --num_envs=1024 \
    --headless \
    --teacher_model=$TEACHER_MODEL
```

### 2. 监控训练
- [ ] 启动 tensorboard

```bash
tensorboard --logdir logs/sirius_curriculum_il
```

- [ ] 检查关键指标 (每100 iterations)
  - [ ] `Train/mean_reward`: 逐步接近教师水平 (目标 75+)
  - [ ] `IL/imitation_loss`: 逐步下降 (目标 < 0.01)
  - [ ] `IL/action_mse`: 学生与教师动作差异 (目标 < 0.01)
  - [ ] `Loss/learning_rate`: 学习率调度正常
  - [ ] GPU利用率 > 80%

### 3. 中期检查 (~1000 iterations)
- [ ] RL reward 达到 60-65
- [ ] IL loss 降到 0.02-0.05
- [ ] 无 NaN 或 Inf 错误
- [ ] 相机增强正常工作

### 4. 阶段1完成检查 (~2000 iterations)
- [ ] RL reward > 75
- [ ] IL loss < 0.01
- [ ] Action MSE < 0.01
- [ ] 模型保存正常

### 5. 阶段1评估
- [ ] 在平地上测试学生策略

```bash
python legged_gym/scripts/play.py \
    --task=sirius_curriculum_il \
    --load_run=logs/sirius_curriculum_il/stage1_flat_parallel
```

- [ ] 对比教师和学生表现
  - [ ] 速度跟随质量
  - [ ] 姿态稳定性
  - [ ] Episode长度
  - [ ] 视觉编码器是否学到有用特征

---

## 🏔️ 阶段2: Curriculum微调

### 1. 启动微调
- [ ] 确认阶段1的checkpoint路径

```bash
export STAGE1_RUN="logs/sirius_curriculum_il/stage1_flat_parallel"
```

- [ ] 运行阶段2训练

```bash
python legged_gym/scripts/train.py \
    --task=sirius_curriculum_finetune \
    --num_envs=512 \
    --headless \
    --resume \
    --load_run=$STAGE1_RUN \
    --checkpoint=-1
```

### 2. 监控训练
- [ ] 检查关键指标 (每100 iterations)
  - [ ] `Train/mean_reward`: 在复杂地形上的表现
  - [ ] `Curriculum/mean_terrain_level`: 地形难度进度 (目标 6+)
  - [ ] `Train/episode_length`: Episode长度变化
  - [ ] 相机增强强度逐步增加

### 3. 中期检查 (~750 iterations)
- [ ] 平均地形难度 > 5
- [ ] RL reward > 65
- [ ] 能在中等难度地形稳定行走
- [ ] 视觉输入对导航起作用

### 4. 阶段2完成检查 (~1500 iterations)
- [ ] 平均地形难度 > 6.5
- [ ] RL reward > 70
- [ ] 能在高难度地形稳定行走
- [ ] 模型保存正常

### 5. 阶段2评估
- [ ] 在curriculum地形上测试

```bash
python legged_gym/scripts/play.py \
    --task=sirius_curriculum_finetune \
    --load_run=logs/sirius_curriculum_il/stage2_curriculum_finetune
```

- [ ] 测试不同地形类型
  - [ ] 斜坡 (上坡/下坡)
  - [ ] 粗糙表面
  - [ ] 台阶
  - [ ] 缝隙/坑洞
  
- [ ] 测试不同难度级别
  - [ ] 低难度 (0-3): 应完全稳定
  - [ ] 中难度 (4-6): 应稳定通过
  - [ ] 高难度 (7-9): 应能通过大部分

---

## 📊 最终验证

### 1. 性能对比
创建对比表格：

| 模型 | 环境 | RL Reward | Episode长度 | 成功率 |
|------|------|-----------|-------------|--------|
| 教师 | 平地 | 80+ | 20+ | 100% |
| 学生(阶段1) | 平地 | 75+ | 18+ | 95%+ |
| 学生(阶段2) | 难度3 | 70+ | 16+ | 90%+ |
| 学生(阶段2) | 难度6 | 65+ | 14+ | 80%+ |

### 2. 视觉验证
- [ ] 保存深度图可视化
- [ ] 检查视觉编码器激活
- [ ] 对比有/无视觉输入的性能差异

### 3. 鲁棒性测试
- [ ] 测试不同速度命令
- [ ] 测试不同相机噪声水平
- [ ] 测试不同地形分布

---

## ❌ 故障排查检查清单

### 问题1: IL损失不下降
- [ ] 检查教师模型是否正确加载: `print(self.alg.teacher_actor_critic)`
- [ ] 检查视觉编码器梯度是否正常
- [ ] 尝试增加学习率: `learning_rate = 1e-3`
- [ ] 尝试降低IL权重: `imitation_loss_coef = 0.5`

### 问题2: RL奖励下降
- [ ] 降低IL损失权重
- [ ] 检查教师策略质量
- [ ] 启用IL权重衰减

### 问题3: GPU内存不足
- [ ] 减少环境数: `--num_envs=512`
- [ ] 增加相机更新间隔: `camera.update_interval = 4`
- [ ] 检查是否有内存泄漏

### 问题4: 训练不稳定
- [ ] 降低学习率
- [ ] 增加梯度裁剪: `max_grad_norm = 0.5`
- [ ] 检查奖励缩放是否合理

---

## ✅ 完成标志

### 阶段1完成 ✓
- [ ] RL reward > 75
- [ ] IL loss < 0.01
- [ ] 平地上稳定行走
- [ ] checkpoint保存完整

### 阶段2完成 ✓
- [ ] RL reward > 70 (复杂地形)
- [ ] 地形课程晋级到难度7+
- [ ] 视觉对导航起关键作用
- [ ] checkpoint保存完整

### 项目完成 ✓✓✓
- [ ] 两个阶段全部完成
- [ ] 性能对比表格完成
- [ ] 视觉验证完成
- [ ] 鲁棒性测试完成
- [ ] 文档整理完成

---

## 📝 记录和报告

### 训练日志
记录每个阶段的关键信息：

**阶段1:**
- 开始时间: ____
- 结束时间: ____
- 最终RL reward: ____
- 最终IL loss: ____
- 遇到的问题: ____

**阶段2:**
- 开始时间: ____
- 结束时间: ____
- 最终RL reward: ____
- 最终地形难度: ____
- 遇到的问题: ____

### 模型路径记录
- 教师模型: `____`
- 阶段1最佳模型: `____`
- 阶段2最佳模型: `____`

---

**预计总时间**: 5-7天
**当前进度**: [ ] 准备阶段 → [ ] 阶段1 → [ ] 阶段2 → [ ] 验证完成
