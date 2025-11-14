# Sirius 模型迁移与视觉IL训练方案

## 📋 目标

将 `task=sirius` (平地，无视觉) 训练的本体感知策略迁移到 `task=sirius_curriculum` (多地形，带视觉) 环境，通过模仿学习(IL)训练视觉-本体感知端到端策略。

## 🏗️ 架构概览

```
阶段1: 教师策略训练 (Teacher Policy)
  task=sirius (平地, 无视觉)
  输入: 本体感知 (45维)
  输出: 动作 (12维)
  ↓
阶段2: 并行训练 (Parallel Training)
  task=sirius_curriculum (多地形, 带视觉)
  教师: 冻结的本体策略
  学生: 视觉-本体融合策略
  损失: RL奖励 + IL蒸馏损失
  ↓
阶段3: 后训练微调 (Fine-tuning)
  task=sirius_curriculum (curriculum地形)
  纯RL训练，适应复杂地形
```

## 📝 详细实施步骤

### 步骤1: 训练教师策略 (已完成？)

```bash
# 如果还没有训练好的 sirius 模型，先训练
python legged_gym/scripts/train.py --task=sirius --num_envs=4096 --headless

# 训练到收敛（~1800 iterations）
# 模型保存在: logs/sirius/YYYY-MM-DD/HH-MM-SS/model_*.pt
```

**验证教师策略质量:**
- 平均奖励 > 80
- 能够稳定跟随速度命令
- 姿态控制良好

### 步骤2: 创建IL训练配置

需要创建一个新的配置文件支持IL训练模式。

#### 2.1 创建 `sirius_curriculum_il_config.py`

这个配置继承自 `sirius_curriculum_config.py`，添加IL相关设置：

```python
# legged_gym/legged_gym/envs/sirius_diff_vis/sirius_curriculum_il_config.py

from .sirius_curriculum_config import SiriusCurriculumCfg, SiriusCurriculumCfgPPO, SiriusCurriculum
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class SiriusCurriculumILCfg(SiriusCurriculumCfg):
    """IL训练的环境配置 - 初期在平地训练"""
    
    class terrain(SiriusCurriculumCfg.terrain):
        # 🔧 IL阶段1: 先在平地训练（与教师策略环境一致）
        mesh_type = "plane"  # 平地
        curriculum = False   # 关闭地形课程
        measure_heights = False  # 平地不需要高度测量
        
    class camera(SiriusCurriculumCfg.camera):
        # 保持相机配置，但初期可以减少数据增强
        augmentation_curriculum = True
        # 数据增强范围更保守（初期）
        noise_std_range = [0.0, 0.02]  # 减少噪声
        dropout_prob_range = [0.0, 0.05]  # 减少dropout


class SiriusCurriculumILCfgPPO(SiriusCurriculumCfgPPO):
    """IL训练的PPO配置"""
    
    class algorithm(SiriusCurriculumCfgPPO.algorithm):
        # IL损失权重
        use_imitation_loss = True
        imitation_loss_coef = 1.0  # IL损失系数（与RL损失平衡）
        
        # 学习率可以稍高（因为有教师指导）
        learning_rate = 5e-4
        
        # 熵系数可以降低（减少探索，更多模仿）
        entropy_coef = 0.01
        
    class runner(SiriusCurriculumCfgPPO.runner):
        experiment_name = "sirius_curriculum_il"
        run_name = "stage1_flat_parallel"
        
        # 教师模型路径（从 sirius 训练获得）
        teacher_model_path = "logs/sirius/YYYY-MM-DD/HH-MM-SS/model_XXXX.pt"
        
        # IL训练迭代
        max_iterations = 2000  # 阶段1: 平地并行训练
        
        # 保存间隔
        save_interval = 50
```

#### 2.2 修改 PPO 算法支持 IL

需要修改 `rsl_rl/rsl_rl/algorithms/ppo.py` 添加IL损失：

```python
# 在 PPO.update() 方法中添加IL损失计算

def update(self):
    # ... 原有代码 ...
    
    # 计算actor损失
    actions_log_prob_batch = self.actor_critic.act(
        obs_batch, critic_obs_batch
    )[1]
    
    # 原有PPO损失
    ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
    surrogate = -advantages_batch * ratio
    surrogate_clipped = -advantages_batch * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
    )
    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
    
    # IL损失（如果启用）
    if hasattr(self, 'teacher_actor_critic') and self.cfg.use_imitation_loss:
        # 教师策略的动作（仅使用本体观测）
        with torch.no_grad():
            teacher_actions = self.teacher_actor_critic.act_inference(
                obs_batch  # 教师只看本体观测
            )
        
        # 学生策略的动作
        student_actions = self.actor_critic.act_inference(
            obs_batch, 
            depth_obs=depth_obs_batch  # 学生看视觉+本体
        )
        
        # MSE损失（动作匹配）
        imitation_loss = F.mse_loss(student_actions, teacher_actions)
        
        # 总actor损失
        actor_loss = (surrogate_loss 
                     + self.cfg.imitation_loss_coef * imitation_loss
                     - self.entropy_coef * entropy)
    else:
        actor_loss = surrogate_loss - self.entropy_coef * entropy
    
    # ... 后续代码 ...
```

#### 2.3 修改 OnPolicyRunner 加载教师模型

```python
# rsl_rl/rsl_rl/runners/on_policy_runner.py

class OnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device='cpu'):
        # ... 原有代码 ...
        
        # 加载教师模型（如果配置了）
        if hasattr(train_cfg.runner, 'teacher_model_path'):
            print(f"Loading teacher model from: {train_cfg.runner.teacher_model_path}")
            self.load_teacher_model(train_cfg.runner.teacher_model_path)
    
    def load_teacher_model(self, model_path):
        """加载教师策略模型（冻结参数）"""
        from rsl_rl.modules import ActorCritic
        
        # 创建教师网络（仅本体感知，不含视觉编码器）
        teacher_cfg = copy.deepcopy(self.cfg.policy)
        teacher_cfg.use_vision = False  # 教师不使用视觉
        
        self.teacher_actor_critic = ActorCritic(
            self.env.cfg.env.num_observations,
            self.env.cfg.env.num_observations,  # 教师的critic obs = actor obs
            self.env.cfg.env.num_actions,
            **teacher_cfg
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        self.teacher_actor_critic.load_state_dict(checkpoint['model_state_dict'])
        
        # 冻结教师参数
        for param in self.teacher_actor_critic.parameters():
            param.requires_grad = False
        
        self.teacher_actor_critic.eval()
        
        # 将教师传递给算法
        self.alg.teacher_actor_critic = self.teacher_actor_critic
        
        print("✅ Teacher model loaded and frozen")
```

### 步骤3: 阶段1 - 平地并行训练

```bash
# 在平地上进行 RL + IL 并行训练
# 学生策略同时学习：
#   1. 从RL奖励中学习运动控制
#   2. 从教师策略中学习如何将视觉映射到动作

python legged_gym/scripts/train.py \
    --task=sirius_curriculum_il \
    --num_envs=1024 \
    --headless \
    --teacher_model=logs/sirius/YYYY-MM-DD/HH-MM-SS/model_1800.pt

# 训练 ~2000 iterations
# 监控指标：
#   - RL reward 应接近教师水平
#   - IL loss 应逐渐下降
#   - 视觉编码器应开始学习有用特征
```

**训练监控:**
```python
# 在 tensorboard 中查看：
# - RL/mean_reward: 应达到教师水平 (>80)
# - IL/imitation_loss: 应下降到 < 0.01
# - IL/action_mse: 学生与教师动作差异
```

### 步骤4: 阶段2 - Curriculum后训练

创建配置用于后训练阶段：

```python
class SiriusCurriculumFinetuneCfg(SiriusCurriculumCfg):
    """后训练配置 - 启用完整curriculum"""
    
    class terrain(SiriusCurriculumCfg.terrain):
        mesh_type = "trimesh"  # 完整地形
        curriculum = True      # 启用课程学习
        measure_heights = True
        
        # 🔧 从较低难度开始（因为已有基础策略）
        max_init_terrain_level = 3  # 从难度3开始（跳过最简单的）
        
    class camera(SiriusCurriculumCfg.camera):
        # 启用完整数据增强
        augmentation_curriculum = True
        noise_std_range = [0.0, 0.05]
        dropout_prob_range = [0.0, 0.15]


class SiriusCurriculumFinetuneCfgPPO(SiriusCurriculumCfgPPO):
    class algorithm(SiriusCurriculumCfgPPO.algorithm):
        # 关闭IL损失，纯RL训练
        use_imitation_loss = False
        
        # 恢复正常的熵系数（鼓励探索新地形）
        entropy_coef = 0.015
        
        # 学习率可以降低（微调阶段）
        learning_rate = 1e-4
    
    class runner(SiriusCurriculumCfgPPO.runner):
        experiment_name = "sirius_curriculum_il"
        run_name = "stage2_curriculum_finetune"
        
        # 从阶段1的checkpoint继续
        resume = True
        load_run = "logs/sirius_curriculum_il/stage1_flat_parallel"
        checkpoint = -1  # 最新checkpoint
        
        # 后训练迭代
        max_iterations = 1500
```

训练命令：

```bash
# 从阶段1的checkpoint开始，在curriculum地形上微调
python legged_gym/scripts/train.py \
    --task=sirius_curriculum_finetune \
    --num_envs=512 \
    --headless \
    --resume \
    --load_run=logs/sirius_curriculum_il/stage1_flat_parallel

# 训练 ~1500 iterations
# 监控地形课程进度和奖励变化
```

### 步骤5: 评估和对比

```bash
# 评估最终模型
python legged_gym/scripts/play.py \
    --task=sirius_curriculum_finetune \
    --load_run=logs/sirius_curriculum_il/stage2_curriculum_finetune

# 对比测试：
# 1. 教师策略（无视觉）在平地上的表现
# 2. 学生策略（有视觉）在平地上的表现
# 3. 学生策略在curriculum地形上的表现
```

## 📊 预期结果

### 阶段1完成后（平地并行训练）
- ✅ 学生策略在平地上性能接近教师（RL reward > 75）
- ✅ IL损失收敛到低值（< 0.01）
- ✅ 视觉编码器开始提取有用特征（深度图中的障碍物信息）

### 阶段2完成后（Curriculum微调）
- ✅ 学生策略能在多种地形上稳定行走
- ✅ 地形课程能够逐步晋级到高难度
- ✅ 视觉输入对复杂地形导航起到关键作用

## 🔧 关键技术点

### 1. 网络架构一致性
确保教师和学生的本体感知分支结构一致：
- 教师: 本体 (45) → MLP [256, 128, 64] → 动作 (12)
- 学生: 本体 (45) → MLP [256, 128, 64] ↘
         视觉 (58×87) → CNN → FiLM ↗ → 融合 → 动作 (12)

### 2. IL损失权重调节
- 初期: `imitation_loss_coef = 1.0` (强依赖教师)
- 中期: `imitation_loss_coef = 0.5` (平衡IL和RL)
- 后期: `imitation_loss_coef = 0.0` (纯RL，阶段2)

### 3. 数据增强课程
随训练进度逐步增加相机噪声，提高鲁棒性：
- 阶段1（平地）：轻度增强
- 阶段2（curriculum）：完整增强

### 4. 地形课程策略
- 阶段1：固定平地 (mesh_type="plane")
- 阶段2：从中等难度开始 (max_init_terrain_level=3)
- 避免从零开始curriculum（因为已有基础能力）

## 📁 文件结构

```
legged_gym/legged_gym/envs/sirius_diff_vis/
├── sirius_flat_config.py              # 教师策略环境（已有）
├── sirius_curriculum_config.py        # Curriculum基础配置（已有）
├── sirius_curriculum_il_config.py     # IL训练配置（新建）
└── sirius_curriculum_finetune_config.py  # 后训练配置（新建）

rsl_rl/rsl_rl/
├── algorithms/
│   └── ppo.py                         # 修改：添加IL损失
└── runners/
    └── on_policy_runner.py            # 修改：加载教师模型

legged_gym/legged_gym/envs/__init__.py  # 注册新任务
```

## ⚠️ 注意事项

1. **教师模型质量**: 确保教师策略训练充分（平地reward > 80）
2. **观测一致性**: 教师和学生的本体观测维度必须完全一致（45维）
3. **视觉编码器初始化**: 确保CNN权重合理初始化（Kaiming初始化）
4. **IL损失平衡**: 监控IL loss和RL reward的比例，避免过度拟合教师
5. **内存使用**: 并行训练会同时加载教师和学生模型，注意GPU内存
6. **地形跳过**: 保留 `skip_terrain_types=[4,5]` 避免难以学习的地形

## 🚀 快速启动检查清单

- [ ] 1. 确认教师模型已训练完成并保存
- [ ] 2. 创建 `sirius_curriculum_il_config.py`
- [ ] 3. 修改 `ppo.py` 添加IL损失计算
- [ ] 4. 修改 `on_policy_runner.py` 加载教师模型
- [ ] 5. 在 `__init__.py` 注册新任务
- [ ] 6. 运行阶段1训练（平地并行）
- [ ] 7. 监控训练指标（RL + IL）
- [ ] 8. 创建 `sirius_curriculum_finetune_config.py`
- [ ] 9. 运行阶段2训练（Curriculum微调）
- [ ] 10. 评估最终模型性能

## 📚 参考资料

- **Learning to Walk in Minutes Using Massively Parallel Deep RL** (Isaac Gym论文)
- **Rapid Locomotion via Reinforcement Learning** (ETH苏黎世，ANYmal)
- **Walk These Ways: Tuning Robot Control for Generalization** (视觉导航)

---

**预计时间线:**
- 阶段1（平地并行）: 2-3天训练 + 1天调试
- 阶段2（Curriculum微调）: 1-2天训练
- 总计: 约5-7天

**硬件需求:**
- GPU: RTX 3090 或更好（24GB显存推荐）
- CPU: 16核以上
- RAM: 32GB+
