# 🎓 IL训练启动指南 - 视觉模型学习教师策略

## ✅ 已完成配置

### 1. 教师模型
- **路径**: `/home/eziothean/ATEC_VIS_E2E/legged_gym/logs/sirius_teacher_curriculum/Nov14_22-58-38_/model_2250.pt`
- **类型**: 45维本体感知MLP策略
- **训练环境**: Curriculum地形 + heading_command模式

### 2. IL配置更新
已修改 `sirius_curriculum_il_config.py`:
- ✅ **地形**: 使用curriculum地形（与教师一致）
- ✅ **命令采样**: 使用heading_command模式（朝向对齐）
- ✅ **命令课程**: 与教师相同的课程设置
- ✅ **教师模型路径**: 已硬编码到配置中
- ✅ **命令采样策略**: 完全与教师一致（包括横向速度限制）

### 3. 基类更新
已修改 `sirius_curriculum_config.py`:
- ✅ 添加 `heading_command = True` 到配置
- ✅ 实现 `_post_physics_step_callback`（支持heading转换）
- ✅ 实现 `_resample_commands`（朝向对齐 + 横向限制）
- ✅ `SiriusCurriculumIL` 正确继承所有功能

## 🚀 启动IL训练

### 命令
```bash
cd /home/eziothean/ATEC_VIS_E2E/legged_gym/legged_gym/scripts

# 激活conda环境
conda activate sirius2

# 启动训练（建议1024个环境，IL训练效率高）
python train.py --task=sirius_curriculum_il --num_envs=1024 --headless
```

### 预期行为
训练开始时应该看到：
```
🔥 [SiriusCurriculumIL] IL Training Environment initialized
  - Terrain: trimesh (curriculum=True)
  - Camera: True
  - Num envs: 1024
  - heading_command: True
  - Command curriculum: True
```

## 📊 训练细节

### IL策略
- **模式**: RL + IL并行训练
- **IL损失系数**: 1.0 → 0.3（1500次迭代线性衰减）
- **学习率**: 5e-4（比纯RL稍高）
- **熵系数**: 0.01（降低随机探索，更多模仿）

### 学生策略
- **类型**: 视觉-本体融合（VisionProprioceptionActorCritic）
- **视觉编码器**: FiLM-ResNet (87x58深度图)
- **相机增强**: 渐进式（跟随地形难度）

### 训练参数
- **max_iterations**: 2000
- **num_steps_per_env**: 24
- **num_learning_epochs**: 8
- **save_interval**: 50

## 🔍 监控指标

### 关键指标
1. **IL Loss**: 动作匹配损失，应该逐渐下降
2. **RL Rewards**: 
   - `tracking_lin_vel`: 线速度跟踪（权重15）
   - `tracking_ang_vel`: 角速度跟踪（权重10）
3. **Command Curriculum**: 速度范围应逐步扩展
4. **Terrain Curriculum**: 地形难度应逐步提升

### Tensorboard
```bash
tensorboard --logdir=legged_gym/logs/sirius_curriculum_il
```

## 📁 输出位置
- **日志**: `legged_gym/logs/sirius_curriculum_il/<timestamp>/`
- **模型**: `legged_gym/logs/sirius_curriculum_il/<timestamp>/model_*.pt`
- **Tensorboard**: `legged_gym/logs/sirius_curriculum_il/<timestamp>/summaries/`

## ⚙️ 关键配置参数

### 可以调整的参数
如果训练效果不理想，可以修改：

1. **IL损失权重** (`sirius_curriculum_il_config.py`):
   ```python
   imitation_loss_coef = 1.0  # 增大 = 更多模仿，减小 = 更多RL探索
   ```

2. **学习率**:
   ```python
   learning_rate = 5e-4  # 降低 = 更稳定，提高 = 更快收敛
   ```

3. **相机增强强度** (如果视觉学习困难):
   ```python
   noise_std_range = [0.0, 0.02]  # 减小范围 = 降低难度
   ```

## 🎯 下一步：微调（Finetune）

IL训练完成后（~2000次迭代），可以进行纯RL微调：
```bash
python train.py --task=sirius_curriculum_finetune --num_envs=1024 --headless \
    --resume --load_run=logs/sirius_curriculum_il/<timestamp>
```

微调阶段会：
- ❌ 关闭IL损失（纯RL）
- ✅ 保持视觉-本体融合
- ✅ 继续在curriculum地形上训练
- 🎯 优化策略以适应更复杂地形

## 🐛 故障排查

### 如果IL Loss不下降
- 检查教师模型路径是否正确
- 增加`imitation_loss_coef`
- 检查学生和教师的观测是否对齐

### 如果RL Rewards很低
- 降低`imitation_loss_coef`（给RL更多权重）
- 检查命令采样是否正确（heading_command模式）
- 检查地形课程是否太难（降低初始难度）

### 如果出现"螃蟹步"
- 检查`heading_command = True`是否生效
- 检查`_resample_commands`是否正确覆盖
- 验证横向速度是否被正确限制（应该≤30%前进速度）
