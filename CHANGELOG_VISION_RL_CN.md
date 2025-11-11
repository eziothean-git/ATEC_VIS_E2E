# 视觉-本体融合 RL 系统实现 - 更新日志

**日期**: 2025-11-11  
**分支**: CNN  
**作者**: GitHub Copilot & User

## 概述

实现了完整的视觉-本体融合强化学习系统，将深度图像观测集成到 Sirius 四足机器人的 PPO 训练流程中。

## 架构设计

### 数据流
```
环境 (Isaac Gym)
  ├─ obs_buf (45维本体观测)
  └─ depth_obs_buf (1x58x87 深度图)
       ↓
OnPolicyRunner
  ├─ PPO.act(obs, depth_obs)
  └─ RolloutStorage (存储 obs + depth)
       ↓
VisionProprioceptionActorCritic
  ├─ SimpleCNNEncoder: depth → 32维特征
  └─ MLP: cat(proprio_45, vision_32) = 77维 → actions
```

### 网络结构
- **SimpleCNNEncoder**: 3层CNN (1→16→32→32 channels)
  - 输入: (B, 1, 58, 87) 归一化深度图
  - 输出: (B, 32) 视觉特征
  - 参数: ~29,696
- **Actor/Critic MLP**: [256, 128, 64]
  - 输入: 77维 (45本体 + 32视觉)
  - 总参数: ~152,793

## 修改的文件

### 1. 核心模块实现

#### `rsl_rl/modules/vision_encoder.py` (新建)
- 实现 `SimpleCNNEncoder` 类
- 3层卷积网络 + MaxPooling
- 支持动态输入尺寸计算

#### `rsl_rl/modules/vision_actor_critic.py` (新建)
- 实现 `VisionProprioceptionActorCritic` 类
- 融合本体观测和视觉特征
- 兼容 PPO 的 masks 和 hidden_states 参数

#### `rsl_rl/modules/__init__.py` (修改)
- 导出新的视觉模块类

### 2. 算法层修改

#### `rsl_rl/storage/rollout_storage.py`
**变更**:
- 添加 `depth_images` buffer: `torch.Size([num_steps, num_envs, 1, H, W])`
- 修改 `__init__`: 接受 `depth_image_shape` 参数
- 修改 `add_transitions`: 存储深度图像
- 修改 `mini_batch_generator`: 返回深度图像批次

**影响**: 支持深度图像的经验回放

#### `rsl_rl/algorithms/ppo.py`
**变更**:
- 修改 `act()`: 接受 `depth_obs` 参数，运行时类型检查
- 修改 `compute_returns()`: 传递 `depth_obs` 给 critic
- 修改 `update()`: 从 storage 获取 `depth_obs_batch` 并传递给策略

**影响**: PPO 算法支持视觉输入

#### `rsl_rl/runners/on_policy_runner.py`
**变更**:
- 添加 `VisionEncoderCfg` 包装类 (dict→object 转换)
- 修改 `__init__`: 
  - 检测 `use_vision` 标志
  - 根据类型创建不同的 actor_critic
  - 初始化带深度图像的 RolloutStorage
- 修改 `learn()`: 从环境获取 `depth_obs_buf` 并传递给 PPO

**影响**: 训练循环协调视觉数据流

### 3. 环境层修改

#### `legged_gym/envs/sirius_diff_vis/sirius_joystick.py`
**变更**:
- 添加 `depth_obs_buf` 缓冲区
- 修改 `compute_observations()`:
  - 调用 `get_camera_depth_images()` 获取深度图
  - 归一化到 [0,1]
  - 添加通道维度 → (B, 1, H, W)

**影响**: 环境输出深度观测

#### `legged_gym/envs/base/base_task.py`
**变更**:
- 修改 headless 模式下的图形设备逻辑:
  - 检查 `cfg.camera.enable` 标志
  - 如果相机启用，即使在 headless 模式也保持 `graphics_device_id` 激活
  - 允许在无窗口情况下获取深度图

**影响**: 修复 headless 训练时的相机 bug

### 4. 配置文件修改

#### `legged_gym/envs/sirius_diff_vis/sirius_shared_model.py`
**变更**:
- 添加 `vision_encoder` 配置类:
  - CNN 层定义: 3层，输出 32 维
  - 输入尺寸: 58x87
  - 激活函数: ReLU
- 修改 `policy` 类:
  - 设置 `use_vision = True`
  - 设置 `vision_latent_dim = 32`
- 修改 `runner` 类:
  - 设置 `policy_class_name = 'VisionProprioceptionActorCritic'`

**影响**: 两个任务共享相同的视觉模型配置

#### `legged_gym/envs/sirius_diff_vis/sirius_flat_config.py`
**变更**:
- 显式继承 `vision_encoder` 类 (修复 class_to_dict 问题)
- 保持 `num_observations = 45` (只包含本体观测)
- 添加 `algorithm` 覆盖:
  - `learning_rate = 2.5e-4` (原来的 1/4，补偿批量大小从 4096→1024)

**影响**: 平地任务配置完整

#### `legged_gym/envs/sirius_diff_vis/sirius_bridge_env.py`
**变更**:
- 保持 `num_observations = 45`

**影响**: 桥梁任务配置一致

## 关键问题修复

### 问题 1: vision_encoder 配置丢失
**症状**: `ValueError: VisionProprioceptionActorCritic requires vision_encoder config`

**根因**: `class_to_dict()` 只提取当前类的属性，子类没有显式继承 `vision_encoder`

**解决**: 在 `SiriusFlatCfgPPO` 中添加:
```python
class vision_encoder(SiriusSharedPPOCfg.vision_encoder):
    pass
```

### 问题 2: 观测维度不匹配
**症状**: `RuntimeError: The size of tensor a (45) must match the size of tensor b (77)`

**根因**: `num_observations` 被错误设置为 77，但 `obs_buf` 应该只包含 45 维本体观测

**解决**: 将 `num_observations` 改回 45，视觉特征通过 `depth_obs_buf` 单独传递

### 问题 3: 本体观测维度计算错误
**症状**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x77 and 45x256)`

**根因**: `num_proprio_obs = self.env.num_obs - vision_latent_dim` 错误地减去了视觉维度

**解决**: 改为 `num_proprio_obs = self.env.num_obs` (直接使用 45)

### 问题 4: Headless 模式相机失败
**症状**: `Error: could not find camera with handle -1 in environment`

**根因**: Headless 模式下 `graphics_device_id = -1`，导致相机无法创建

**解决**: 检查 `cfg.camera.enable`，如果启用则保持图形设备激活

## 测试结果

### 模块单元测试 (`test_vision_modules.py`)
- ✅ SimpleCNNEncoder: 输入 (16,1,58,87) → 输出 (16,32)
- ✅ VisionProprioceptionActorCritic: 输入 proprio(16,45) + depth(16,1,58,87) → actions(16,12)
- ✅ 大批量测试: 1024 envs 正常工作

### 集成测试
- ⏳ 等待完整训练验证

## 训练配置

### 超参数调整
- **批量大小**: 4096 → 1024 envs (GPU 内存限制)
- **学习率**: 1e-3 → 2.5e-4 (补偿批量大小减少)
- **迭代次数**: 1200 (保持不变)

### 训练命令
```bash
# 平地训练 (阶段1)
python train.py --task=sirius --num_envs=1024 --headless

# 桥梁训练 (阶段2 - 从阶段1 resume)
python train.py --task=sirius_diff_vis --num_envs=1024 --headless --resume
```

## 技术亮点

1. **双模态融合**: 本体感觉 + 视觉特征，充分利用多模态信息
2. **模块化设计**: CNN 编码器独立，易于替换或调整
3. **兼容性**: 保持与标准 ActorCritic 的接口一致性
4. **配置共享**: 两阶段训练使用相同网络结构
5. **运行时类型检查**: 使用 `isinstance()` 避免修改基类

## 文件统计

- **新增文件**: 2 个 (vision_encoder.py, vision_actor_critic.py)
- **修改文件**: 8 个 (PPO, Storage, Runner, Environment, Configs)
- **代码行数**: ~800 行 (包含注释和调试信息)
- **网络参数**: ~152K (Vision: 29K, MLP: 123K)

## 下一步

- [ ] 运行完整训练，验证收敛性
- [ ] 监控 TensorBoard 指标
- [ ] 评估视觉特征的贡献度
- [ ] 考虑添加视觉预训练（如果需要）
- [ ] 测试阶段2的迁移学习效果

## 备注

- 所有修改已通过模块测试
- 代码包含详细的中文注释
- 遵循原有代码风格和架构
- 保持向后兼容性（非视觉任务仍可正常工作）
