# Sirius 配置分离说明

## 🎯 配置策略

为了支持教师-学生模型迁移，我们将配置分为三类：

### 1️⃣ 教师策略 (task=sirius)
- **文件**: `sirius_flat_config.py`
- **环境**: 平地 (plane)
- **相机**: ❌ 关闭 (`camera.enable = False`)
- **策略**: `ActorCritic` (纯本体感知 MLP)
- **输入**: 45维本体观测
- **网络**: MLP [256, 128, 64]
- **用途**: 训练教师策略，不依赖视觉

### 2️⃣ 学生策略 - IL训练 (task=sirius_curriculum_il)
- **文件**: `sirius_curriculum_il_config.py`
- **环境**: 平地 (plane) - 与教师一致
- **相机**: ✅ 启用 (`camera.enable = True`)
- **策略**: `VisionProprioceptionActorCritic` (视觉+本体)
- **输入**: 45维本体观测 + 58×87深度图
- **网络**: CNN + FiLM + MLP [256, 128, 64]
- **用途**: 通过IL从教师学习，同时学习视觉特征

### 3️⃣ 学生策略 - 微调 (task=sirius_curriculum_finetune)
- **文件**: `sirius_curriculum_finetune_config.py`
- **环境**: Curriculum (多地形)
- **相机**: ✅ 启用 + 完整数据增强
- **策略**: `VisionProprioceptionActorCritic`
- **用途**: 在复杂地形上微调，提高泛化能力

## 📊 配置对比表

| 配置 | 任务名 | 地形 | 相机 | 策略类 | 网络结构 |
|------|--------|------|------|--------|----------|
| 教师 | sirius | plane | ❌ | ActorCritic | MLP [256,128,64] |
| 学生IL | sirius_curriculum_il | plane | ✅ | VisionProprioceptionActorCritic | CNN+FiLM+MLP |
| 学生微调 | sirius_curriculum_finetune | curriculum | ✅ | VisionProprioceptionActorCritic | CNN+FiLM+MLP |

## 🔑 关键修改

### `sirius_flat_config.py`

#### 修改前 ❌
```python
class SiriusFlatCfgPPO:
    class vision_encoder(SiriusSharedPPOCfg.vision_encoder):
        pass  # 继承视觉配置
    
    class policy(SiriusSharedPPOCfg.policy):
        pass  # 继承视觉策略
    
    class runner(SiriusSharedPPOCfg.runner):
        policy_class_name = 'VisionProprioceptionActorCritic'  # 视觉策略

class camera:
    enable = True  # 启用相机
```

#### 修改后 ✅
```python
class SiriusFlatCfgPPO:
    # 不继承视觉配置
    
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        # 不设置 use_vision
    
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'  # 纯MLP策略

class camera:
    enable = False  # 关闭相机
```

## 🚀 使用方法

### 阶段0: 训练教师策略
```bash
python train.py --task=sirius --num_envs=4096 --headless

# 预期输出:
# - 使用 ActorCritic (纯MLP)
# - 输入: 45维本体观测
# - 相机: 关闭
# - 训练 ~1800 iterations
```

### 阶段1: IL并行训练
```bash
python train.py --task=sirius_curriculum_il --num_envs=1024 --headless

# 预期输出:
# - 使用 VisionProprioceptionActorCritic
# - 输入: 45维本体 + 58×87深度
# - 相机: 启用
# - 从教师学习
```

### 阶段2: Curriculum微调
```bash
python train.py --task=sirius_curriculum_finetune --num_envs=512 --headless \
    --resume --load_run=logs/sirius_curriculum_il/stage1_flat_parallel

# 预期输出:
# - 使用 VisionProprioceptionActorCritic
# - 地形: Curriculum
# - 相机: 完整增强
# - 纯RL微调
```

## ⚠️ 重要注意事项

1. **网络结构一致性**: 教师和学生的MLP部分必须完全相同 ([256, 128, 64])，这样才能进行IL知识迁移

2. **相机配置**: 
   - 教师训练时必须关闭相机 (`enable = False`)
   - 学生训练时必须启用相机 (`enable = True`)

3. **策略类名称**:
   - 教师: `'ActorCritic'` (标准MLP)
   - 学生: `'VisionProprioceptionActorCritic'` (视觉融合)

4. **观测维度**:
   - 教师: 45维 (仅本体)
   - 学生: 45维本体 + 视觉latent (通过网络融合)

## 📝 验证清单

在训练前确认：

- [ ] `task=sirius` 使用 `ActorCritic`
- [ ] `task=sirius` 相机关闭 (`camera.enable = False`)
- [ ] `task=sirius_curriculum_il` 使用 `VisionProprioceptionActorCritic`
- [ ] `task=sirius_curriculum_il` 相机启用 (`camera.enable = True`)
- [ ] 两者的MLP隐藏层维度相同 ([256, 128, 64])

---

**修改日期**: 2025年11月14日
**目的**: 分离教师策略（纯MLP）和学生策略（视觉融合）的配置
