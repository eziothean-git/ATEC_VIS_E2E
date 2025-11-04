# 深度相机系统说明

## 🎯 功能概述

成功在 Sirius 机器狗上集成了深度相机系统，可以获取实时深度图像数据。

## 📷 相机配置

### 当前配置（在 `sirius_flat_config.py`）

```python
camera.enable = True
camera.body_name = "base"  # 挂载在机身
camera.position = [0.30, 0.0, 0.12]  # 相对body的位置: [前, 左, 上]
camera.rpy = [0.0, -0.3, 0.0]  # 俯仰角-0.3rad ≈ -17°，向下看
camera.width = 320
camera.height = 240
camera.horizontal_fov = 90.0  # 水平视场角90度
```

### 相机位置说明

- **挂载点**: `base` body（机器人主体）
- **位置**: 机身前方30cm，高度12cm，水平居中
- **朝向**: 朝前并向下倾斜17度，可以同时看到前方和地面
- **视野**: 90度水平视场角，覆盖机器人前方广阔区域

## 🎮 使用方法

### 运行 Demo

```bash
# 注意：不要使用 --headless，相机需要图形上下文
conda activate sirius_rl
python legged_gym/scripts/camera_demo.py --task=sirius
```

Demo 会在 Isaac Gym 窗口中显示相机位置的可视化标记：
- 🔴 **红色球体** = 相机位置
- 🔴 **红色箭头** = 相机朝向（forward）
- 🟢 **绿色横线** = 相机左右方向
- 🔵 **蓝色竖线** = 相机向上方向

你可以用鼠标拖拽旋转视角，从不同角度观察相机在机器狗上的安装位置。

### 在代码中使用

```python
from legged_gym.utils.task_registry import task_registry

# 1. 配置相机
env_cfg, _ = task_registry.get_cfgs(name="sirius")
env_cfg.camera.enable = True
env_cfg.camera.body_name = "base"  # 或 "trunk"
env_cfg.camera.position = [0.30, 0.0, 0.12]
env_cfg.camera.rpy = [0.0, -0.3, 0.0]

# 2. 创建环境
env, _ = task_registry.make_env(name="sirius", args=args, env_cfg=env_cfg)

# 3. 获取深度图
depth_images = env.get_camera_depth_images(as_torch=True)
# 返回: torch.Tensor of shape (num_envs, height, width)
# 或
depth_images = env.get_camera_depth_images(as_torch=False)
# 返回: numpy.ndarray of shape (num_envs, height, width)
```

## 📊 深度数据说明

### 深度值含义

- **单位**: 米 (m)
- **有效范围**: 通常 0.07m - 250m
- **-inf 值**: 表示超出相机远平面，或没有检测到物体
- **负值**: Isaac Gym 特殊表示，通常是远距离物体

### 当前性能

根据 demo 运行结果：
- 有效像素比例: 28-36%
- 深度范围: -0.07m 到 -250m 左右
- 分辨率: 320x240 = 76,800 像素/帧

## 📁 生成的文件

Demo 会在 `camera_outputs/` 目录生成以下文件：

- `depth_step_0000.png` - 第0步的深度图
- `depth_step_0030.png` - 第30步的深度图
- ... 每30步保存一张
- `depth_step_0270.png` - 第270步的深度图

每张图包含：
- 4个并排的环境（如果有多个环境）
- 伪彩色深度可视化（turbo colormap）
- 颜色条显示深度刻度

## 🔧 调整相机参数

### 改变相机位置

```python
# 更靠前
env_cfg.camera.position = [0.40, 0.0, 0.12]

# 更高
env_cfg.camera.position = [0.30, 0.0, 0.20]

# 偏左
env_cfg.camera.position = [0.30, 0.10, 0.12]
```

### 改变相机朝向

```python
# 向下看更多（增加俯仰角）
env_cfg.camera.rpy = [0.0, -0.5, 0.0]  # 约-29度

# 水平向前看
env_cfg.camera.rpy = [0.0, 0.0, 0.0]

# 向左看
env_cfg.camera.rpy = [0.0, 0.0, 0.5]  # yaw旋转
```

### 改变分辨率

```python
# 高分辨率
env_cfg.camera.width = 640
env_cfg.camera.height = 480

# 低分辨率（更快）
env_cfg.camera.width = 160
env_cfg.camera.height = 120
```

## ⚠️ 注意事项

1. **必须关闭 headless 模式**: 相机传感器需要图形渲染上下文
   ```bash
   # ✓ 正确
   python script.py --task=sirius
   
   # ✗ 错误（相机创建失败）
   python script.py --task=sirius --headless
   ```

2. **body_name 必须正确**: 确保 body 名称在 URDF 中存在
   - Sirius 可用: `base`, `trunk`, `imu_link`, `RF_hip`, 等

3. **导入顺序**: Isaac Gym 必须在 torch 之前导入（已在代码中处理）

4. **性能考虑**: 相机渲染会降低仿真速度，建议：
   - 使用较低分辨率进行训练
   - 不是每步都读取深度图
   - 多环境并行时减少环境数量

## 📝 文件修改记录

实现深度相机功能修改了以下文件：

1. `legged_gym/envs/base/legged_robot_config.py` - 添加 camera 配置类
2. `legged_gym/envs/base/legged_robot.py` - 添加相机创建和读取方法
3. `legged_gym/envs/sirius_diff_release/sirius_flat_config.py` - Sirius 相机默认配置
4. `legged_gym/envs/sirius_diff_release/sirius_joystick.py` - 继承相机功能
5. `legged_gym/scripts/camera_demo.py` - Demo 脚本

## 🎉 Demo 运行效果

- ✅ Isaac Gym 窗口显示机器狗和场景
- ✅ 可以用鼠标拖拽旋转视角观察
- ✅ 相机随机器人一起移动
- ✅ 实时获取深度数据
- ✅ 保存可视化图像

## 🚀 下一步

可以基于这个系统：
- 训练视觉导航策略
- 实现障碍物检测
- 添加语义分割
- 结合RGB相机（需修改为 IMAGE_COLOR）
- 实现地形感知

---
生成时间: 2025-11-04
