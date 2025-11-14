# Sirius IL训练快速参考

## 🚀 快速开始

### 前提条件
1. ✅ 确保已有训练好的 sirius 教师模型
2. ✅ 记录教师模型路径，例如: `logs/sirius/2024-01-15/12-34-56/model_1800.pt`

### 阶段1: 平地并行训练（IL + RL）

```bash
# 在平地上训练视觉策略，同时学习RL和模仿教师
python legged_gym/scripts/train.py \
    --task=sirius_curriculum_il \
    --num_envs=1024 \
    --headless

# 训练完成后检查：
# - logs/sirius_curriculum_il/stage1_flat_parallel/
# - 查看 tensorboard: RL reward 应接近教师水平
```

**预期结果:**
- 训练 ~2000 iterations (约2-3小时，RTX 3090)
- RL reward 达到 75+ (接近教师的 80+)
- IL loss 下降到 < 0.01
- 视觉编码器开始学习有用特征

### 阶段2: Curriculum地形微调（纯RL）

```bash
# 从阶段1的checkpoint继续，在复杂地形上微调
python legged_gym/scripts/train.py \
    --task=sirius_curriculum_finetune \
    --num_envs=512 \
    --headless \
    --resume \
    --load_run=logs/sirius_curriculum_il/stage1_flat_parallel \
    --checkpoint=-1  # 使用最新checkpoint

# 训练完成后检查：
# - logs/sirius_curriculum_il/stage2_curriculum_finetune/
# - 查看地形课程进度和奖励变化
```

**预期结果:**
- 训练 ~1500 iterations (约1-2小时)
- 地形难度逐步提升
- 在复杂地形上保持稳定性能

### 评估模型

```bash
# 评估阶段1（平地）
python legged_gym/scripts/play.py \
    --task=sirius_curriculum_il \
    --load_run=logs/sirius_curriculum_il/stage1_flat_parallel

# 评估阶段2（curriculum）
python legged_gym/scripts/play.py \
    --task=sirius_curriculum_finetune \
    --load_run=logs/sirius_curriculum_il/stage2_curriculum_finetune
```

## ⚠️ 注意：需要修改代码

### 必要的代码修改

#### 1. 修改 `rsl_rl/rsl_rl/algorithms/ppo.py`

添加IL损失支持。在 `PPO` 类中：

```python
class PPO:
    def __init__(self, ...):
        # ... 原有代码 ...
        self.teacher_actor_critic = None  # 添加这一行
        
    def update(self):
        # ... 原有PPO损失计算 ...
        
        # ========== 添加IL损失 ==========
        if (hasattr(self, 'teacher_actor_critic') and 
            self.teacher_actor_critic is not None and
            hasattr(self.cfg, 'use_imitation_loss') and 
            self.cfg.use_imitation_loss):
            
            # 教师动作（仅本体观测）
            with torch.no_grad():
                teacher_actions = self.teacher_actor_critic.act_inference(obs_batch)
            
            # 学生动作（本体 + 视觉）
            student_actions = self.actor_critic.act_inference(obs_batch)
            
            # MSE损失
            imitation_loss = F.mse_loss(student_actions, teacher_actions)
            
            # IL损失系数调度（可选）
            il_coef = self.cfg.imitation_loss_coef
            if hasattr(self.cfg, 'imitation_curriculum') and self.cfg.imitation_curriculum:
                schedule = self.cfg.imitation_coef_schedule
                progress = min(self.iter / schedule['iterations'], 1.0)
                il_coef = schedule['start'] + (schedule['end'] - schedule['start']) * progress
            
            # 更新actor损失
            actor_loss = actor_loss + il_coef * imitation_loss
            
            # 记录IL损失
            mean_imitation_loss = imitation_loss.item()
        else:
            mean_imitation_loss = 0.0
        
        # ... 继续原有代码 ...
        
        return mean_value_loss, mean_surrogate_loss, mean_imitation_loss  # 修改返回值
```

#### 2. 修改 `rsl_rl/rsl_rl/runners/on_policy_runner.py`

添加教师模型加载功能：

```python
class OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        # ... 原有代码 ...
        
        # 加载教师模型（如果配置了）
        if hasattr(train_cfg.runner, 'teacher_model_path') and train_cfg.runner.teacher_model_path:
            self.load_teacher_model(train_cfg.runner.teacher_model_path)
        
    def load_teacher_model(self, model_path):
        """加载教师策略模型（冻结参数）"""
        import os
        from rsl_rl.modules import ActorCritic
        
        if not os.path.exists(model_path):
            print(f"❌ Teacher model not found: {model_path}")
            raise FileNotFoundError(f"Teacher model not found: {model_path}")
        
        print(f"[Teacher Model] Loading from: {model_path}")
        
        # 创建教师网络（仅本体感知，不含视觉编码器）
        teacher_cfg = {}
        teacher_cfg['init_noise_std'] = 1.0
        teacher_cfg['actor_hidden_dims'] = [256, 128, 64]
        teacher_cfg['critic_hidden_dims'] = [256, 128, 64]
        teacher_cfg['activation'] = 'elu'
        
        self.teacher_actor_critic = ActorCritic(
            self.env.cfg.env.num_observations,
            self.env.cfg.env.num_observations,
            self.env.cfg.env.num_actions,
            **teacher_cfg
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 提取actor_critic的state_dict
        if 'model_state_dict' in checkpoint:
            teacher_state = checkpoint['model_state_dict']
        elif 'actor_critic' in checkpoint:
            teacher_state = checkpoint['actor_critic']
        else:
            teacher_state = checkpoint
        
        self.teacher_actor_critic.load_state_dict(teacher_state)
        
        # 冻结教师参数
        for param in self.teacher_actor_critic.parameters():
            param.requires_grad = False
        
        self.teacher_actor_critic.eval()
        
        # 将教师传递给算法
        self.alg.teacher_actor_critic = self.teacher_actor_critic
        
        print("✅ Teacher model loaded and frozen")
        print(f"   - Num parameters: {sum(p.numel() for p in self.teacher_actor_critic.parameters())}")
    
    def learn(self, ...):
        # ... 在训练循环中记录IL损失 ...
        
        if len(rewbuffer) > 0:
            # ... 原有代码 ...
            
            # 记录IL损失（如果有）
            if hasattr(self.alg, 'teacher_actor_critic') and self.alg.teacher_actor_critic is not None:
                self.writer.add_scalar('IL/imitation_loss', mean_imitation_loss, locs['it'])
```

#### 3. 修改训练脚本 `legged_gym/scripts/train.py`

添加命令行参数支持教师模型路径：

```python
import argparse

parser = argparse.ArgumentParser()
# ... 原有参数 ...
parser.add_argument('--teacher_model', type=str, default=None, 
                   help='Path to teacher model for IL training')
args = parser.parse_args()

# ... 创建环境和runner ...

# 如果指定了教师模型，设置到配置中
if args.teacher_model:
    train_cfg.runner.teacher_model_path = args.teacher_model
    print(f"[IL Training] Using teacher model: {args.teacher_model}")
```

## 📊 监控指标

### Tensorboard 指标

```bash
# 启动tensorboard
tensorboard --logdir logs/sirius_curriculum_il
```

**阶段1关键指标:**
- `Train/mean_reward`: 应达到 75+ (接近教师的 80+)
- `IL/imitation_loss`: 应下降到 < 0.01
- `IL/action_mse`: 学生与教师动作差异
- `Policy/mean_noise_std`: 探索噪声水平
- `Loss/learning_rate`: 学习率调度

**阶段2关键指标:**
- `Train/mean_reward`: 在curriculum地形上的平均奖励
- `Curriculum/mean_terrain_level`: 平均地形难度（应逐步增加）
- `Train/episode_length`: Episode长度（复杂地形可能更短）

## 🔍 故障排查

### 问题1: IL损失不下降

**症状:** IL loss 一直保持在 0.1 以上

**可能原因:**
- 视觉编码器初始化不当
- 学习率过低
- 教师模型加载失败

**解决方案:**
```python
# 检查教师模型是否正确加载
print(self.alg.teacher_actor_critic)

# 增加学习率
learning_rate = 1e-3  # 从 5e-4 提高

# 检查视觉编码器的梯度
for name, param in self.actor_critic.named_parameters():
    if 'vision' in name:
        print(f"{name}: grad_norm={param.grad.norm()}")
```

### 问题2: RL奖励下降

**症状:** 启用IL后，RL reward 反而低于不用IL的情况

**可能原因:**
- IL损失权重过大，压制了RL信号
- 教师策略本身质量不高

**解决方案:**
```python
# 降低IL损失权重
imitation_loss_coef = 0.5  # 从 1.0 降低

# 或使用渐进式权重衰减
imitation_curriculum = True
imitation_coef_schedule = {
    'start': 1.0,
    'end': 0.1,  # 逐步降低IL依赖
    'iterations': 1000,
}
```

### 问题3: GPU内存不足

**症状:** CUDA out of memory

**可能原因:**
- 同时加载教师和学生模型
- num_envs 过多

**解决方案:**
```bash
# 减少环境数量
--num_envs=512  # 从 1024 降低

# 或增加相机更新间隔
camera.update_interval = 4  # 从 2 增加到 4
```

## 📈 预期训练曲线

### 阶段1（平地IL）
```
Iteration    RL Reward    IL Loss    Action MSE
-------------------------------------------------
0            20-30        0.10       0.15
100          40-50        0.05       0.08
500          60-65        0.02       0.03
1000         70-75        0.01       0.01
2000         75-80        0.005      0.005
```

### 阶段2（Curriculum）
```
Iteration    RL Reward    Terrain Level    Episode Length
----------------------------------------------------------
0            65-70        3.0              15-18
300          60-65        4.5              14-16
600          65-70        5.5              16-18
1000         70-75        6.5              18-20
1500         75-80        7.5              20-22
```

## 🎯 成功标准

### 阶段1完成标准
- [x] RL reward > 75
- [x] IL loss < 0.01
- [x] Action MSE < 0.01
- [x] 平地上稳定跟随速度命令

### 阶段2完成标准
- [x] RL reward > 70 (在难度6+的地形上)
- [x] 地形课程能晋级到难度7+
- [x] 在复杂地形（斜坡、台阶、缝隙）上稳定行走
- [x] 视觉输入对导航起到明显作用

## 📞 联系和支持

如遇到问题，请检查：
1. 教师模型路径是否正确
2. PPO和Runner的代码修改是否正确应用
3. GPU内存是否充足
4. Tensorboard日志是否正常记录

参考完整方案文档: `MODEL_TRANSFER_IL_PLAN.md`
