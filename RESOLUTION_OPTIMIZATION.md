# 降低分辨率配置（可选优化）

## 如果需要支持更多环境，可以降低相机分辨率

### 修改 1：更新配置文件

```python
# legged_gym/envs/sirius_diff_vis/sirius_flat_config.py

class SiriusFlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        # ... 其他配置 ...
        
        # 降低分辨率
        depth_obs_width = 58   # 从 87 改为 58
        depth_obs_height = 43  # 从 58 改为 43
        
        # 更新 num_observations
        num_observations = proprio_obs_dim + depth_obs_width * depth_obs_height
        # = 45 + 58 × 43 = 45 + 2494 = 2539

    class camera(LeggedRobotCfg.camera):
        # ... 其他配置 ...
        width = 58   # 从 87 改为 58
        height = 43  # 从 58 改为 43

class SiriusFlatCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        # 更新 vision_shape
        vision_shape = (1, 43, 58)  # 从 (1, 58, 87) 改为 (1, 43, 58)
```

### 修改 2：更新 ActorCritic（如果已实现 CNN）

```python
# rsl_rl/modules/actor_critic.py

class ActorCritic(nn.Module):
    def __init__(self, ...):
        # CNN 输入会自动适配新的分辨率
        # 如果有硬编码的输出尺寸，需要重新计算
        
        # 例如：
        # 旧：Conv → Pool → Pool 后输出 14×21
        # 新：Conv → Pool → Pool 后输出 10×14
        # 需要更新 Linear 层的输入维度
```

## 效果对比

| 配置 | 分辨率 | 像素数 | 相机显存 (64个) | 支持环境数 | 总显存 (1024 envs) |
|------|--------|--------|----------------|------------|-------------------|
| 高清 | 87×58  | 5046   | 2.5 GB         | 1024       | 8.0 GB            |
| 标清 | 58×43  | 2494   | 1.6 GB         | 2048       | 6.6 GB            |
| 低清 | 43×29  | 1247   | 1.0 GB         | 4096       | 5.5 GB            |

## 降低分辨率的影响

### 优势：
- ✅ **显存大幅降低**：58×43 节省 0.9 GB（64 cameras）
- ✅ **支持更多环境**：可以跑 2048 envs
- ✅ **CNN 计算更快**：卷积层计算量减半
- ✅ **推理速度提升**：~20-30%

### 劣势：
- ⚠️ **视觉细节损失**：远处小障碍物可能看不清
- ⚠️ **特征质量下降**：CNN 提取的特征更粗糙
- ⚠️ **需要重新训练**：不能直接加载旧模型

### 适用场景：
- ✅ 大型障碍物（桥梁木板）：87×58 → 58×43 影响很小
- ✅ 开阔地形：分辨率要求不高
- ⚠️ 复杂地形（小石头、台阶）：可能需要高分辨率
- ❌ 精细操作（抓取、开门）：不建议降低

## 测试建议

### 1. 先在高分辨率下验证
```bash
# 使用 87×58，确保算法有效
python train.py --task=sirius --num_envs=512
```

### 2. 对比测试
```bash
# 修改配置为 58×43
python train.py --task=sirius --num_envs=1024
```

### 3. 评估性能
观察：
- 训练速度（iterations/min）
- 最终奖励值
- 视频中的行为质量

### 4. A/B 测试结果
如果 58×43 的最终性能 ≥ 87×58 的 95%，则可以采用低分辨率。

## 推荐配置组合

### 方案 A：高质量（当前）
```python
max_envs = 64
width = 87, height = 58
obs_refresh_interval = 3
num_envs = 1024
```
- 显存：~8 GB
- 质量：高
- 速度：中等

### 方案 B：平衡型（推荐）
```python
max_envs = 64
width = 58, height = 43  # 降低分辨率
obs_refresh_interval = 4
num_envs = 2048
```
- 显存：~6.6 GB
- 质量：中等
- 速度：快

### 方案 C：极限环境数
```python
max_envs = 64
width = 43, height = 29  # 大幅降低
obs_refresh_interval = 5
num_envs = 4096
```
- 显存：~5.5 GB
- 质量：较低
- 速度：很快

## 实施步骤

如果决定降低分辨率：

1. **备份当前配置**
   ```bash
   cp sirius_flat_config.py sirius_flat_config_backup.py
   ```

2. **修改配置**（见上文"修改 1"）

3. **测试观测维度**
   ```python
   python -c "
   from legged_gym.envs import *
   env = task_registry.make_env('sirius')
   print(f'obs shape: {env.obs_buf.shape}')
   # 应该是 [num_envs, 2539]
   "
   ```

4. **训练测试**
   ```bash
   python train.py --task=sirius --num_envs=1024 --max_iterations=100
   ```

5. **可视化检查**
   ```bash
   # 保存深度图像查看分辨率是否足够
   # 在 sirius_flat_config.py 中临时启用：
   # class camera:
   #     debug_outputs = True
   ```

## 何时应该降低分辨率？

### ✅ 建议降低：
- 16GB 显存想跑 2048+ envs
- 训练速度是瓶颈
- 任务不需要精细视觉（如平地行走）

### ❌ 不建议降低：
- 显存充足（有 24GB+ GPU）
- 需要识别远处小障碍物
- 已经在高分辨率下训练了很久（不想重新开始）
