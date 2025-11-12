# sirius_curriculum_config.py
"""
Sirius 课程学习配置 - 使用官方 terrain.py 的 curriculum 模式训练泛化能力

🎯 课程学习策略：从简单到困难，循序渐进！

课程学习布局：
- Rows (难度): 从易到难，10个级别 (0.0 → 0.9)
- Cols (地形类型): 8种不同障碍类型
  - Col 0-1: 斜坡 (上坡/下坡) - 最简单
  - Col 2: 斜坡 + 粗糙表面
  - Col 3-4: 台阶 (上升/下降)
  - Col 5: 离散障碍物
  - Col 6: 踏脚石
  - Col 7-8: 缝隙/坑洞 - 最困难

训练策略（渐进式）：
1. 🌱 初始阶段（难度 0.0-0.2）：
   - 机器人从最简单的地形开始（难度级别 0-2）
   - 主要是平缓斜坡和小台阶
   - 速度命令较低（0.5 m/s 以内）
   - 🎥 相机：几乎无增强，干净的深度图
   
2. 🌿 成长阶段（难度 0.2-0.5）：
   - 成功后自动晋级到中等难度
   - 引入粗糙表面和较大台阶
   - 速度命令逐步增加
   - 🎥 相机：引入轻度噪声和dropout（<5%）
   
3. 🌳 挑战阶段（难度 0.5-0.9）：
   - 最终达到高难度地形
   - 包含障碍物、踏脚石、缝隙等复杂地形
   - 速度命令达到最大范围
   - 🎥 相机：应用全部增强（噪声、dropout、条带、模糊等）
   
4. 🏆 泛化能力：
   - 最终在所有地形类型和难度级别上都能泛化
   - 对真实世界传感器缺陷具有鲁棒性

📷 相机数据增强课程（Camera Augmentation Curriculum）：
随着地形难度增加，逐步引入以下数据增强，模拟真实世界传感器问题：
- 高斯噪声：0% → 5% 标准差（传感器噪声）
- 像素Dropout：0% → 15%（遮挡、像素失效）
- 随机条带：0% → 10%（滚动快门、扫描线故障）
- 椒盐噪声：0% → 8%（传感器坏点）
- 深度量化：256级 → 64级（有限精度）
- 模糊效果：0 → 3x3核（失焦、运动模糊）
- 亮度/对比度：0.8-1.2x随机变化（光照变化）

使用方法：
    python train.py --task=sirius_curriculum --num_envs=1024 --headless
"""

import torch
from isaacgym import gymtorch
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.sirius_diff_vis.sirius_flat_config import SiriusFlatCfg
from legged_gym.envs.sirius_diff_vis.sirius_joystick import SiriusJoyFlat
from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg


class SiriusCurriculumCfg(SiriusFlatCfg):
    """
    Sirius 课程学习环境配置
    
    使用官方 terrain.py 的 curriculum 模式，提供多样化的地形训练
    """
    
    class env(SiriusFlatCfg.env):
        num_observations = 45  # 本体观测（不包含高度测量，保证部署一致性）
        episode_length_s = 20.0  # 🎯 缩短episode，加快curriculum迭代和学习循环
        num_envs = 1024  # 建议用较多envs覆盖更多地形

    class terrain(SiriusFlatCfg.terrain):
        mesh_type = "trimesh"  # 必须用 trimesh 或 heightfield
        
        # ⚠️ 启用课程学习！
        curriculum = True
        selected = False
        
        # 是否测量高度（必须启用！高度奖励依赖此功能）
        measure_heights = True  # 启用地形高度测量（课程学习必需！）
        
        # 地形网格布局
        terrain_length = 8.0  # 每个子地形的长度（米）
        terrain_width = 8.0   # 每个子地形的宽度（米）
        num_rows = 10         # 难度级别数量（10个难度：0.0 → 0.9，更细粒度的进阶）
        num_cols = 8          # 地形类型数量（8种地形）- 修正：make_terrain实际只有8种
        
        # 🎯 课程学习关键：从最低难度开始！
        max_init_terrain_level = 2  # 最大初始难度级别（索引0-1，对应难度0.0-0.1）
                                     # 机器人将从简单地形开始，逐步晋级
        
        # 🎨 视觉多样性增强：在简单课程的同时，保留少量环境在复杂地形做"视觉探索"
        # 这样视觉编码器从一开始就能见到各种场景，避免过拟合到平地
        visual_exploration_ratio = 0.15  # 15% 的环境用于视觉探索（在所有难度随机分布）
        
        # 地形分辨率
        horizontal_scale = 0.1  # 0.1m per pixel
        vertical_scale = 0.005  # 5mm per unit
        border_size = 20.0      # 边界大小（米）
        
        # 地形类型比例（对应 make_terrain 中的8种类型）
        # [斜坡, 斜坡+粗糙, 台阶上, 台阶下, 障碍物, 踏脚石, 缝隙, 坑洞]
        # 🎯 修改：从简单地形开始，逐步引入复杂地形
        # 比例从低到高：简单地形更常见
        terrain_proportions = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
        
        # 斜率阈值（trimesh生成用）
        slope_treshold = 0.75
        
        # 摩擦系数
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
    class rewards(SiriusFlatCfg.rewards):
        # 课程学习模式下的奖励调整
        class scales:
            # 基础运动奖励
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            
            # 稳定性奖励（对复杂地形很重要）
            orientation = -0.75          # 惩罚倾斜
            base_height = -0.1          # 保持高度（可选）
            
            # 平滑性奖励
            lin_vel_z = -0.075            # 惩罚垂直速度
            ang_vel_xy = -0.05          # 惩罚横滚/俯仰角速度
            collision = -1.0            # 惩罚碰撞
            action_rate = -0.01         # 惩罚动作变化率
            
            # 步态奖励
            feet_air_time = 1.0         # 奖励腾空时间
            stumble = -2.5              # 惩罚绊倒（对崎岖地形重要）
            stand_still = -0.25          # 惩罚原地不动
            
            # 终止惩罚
            termination = -5.0

            orientation = -1.5       # 强姿态惩罚（与 flat 一致）
            feet_air_time = 2      # 腾空奖励（与 flat 一致）
            base_height = -2.5     # 强高度惩罚（与 flat 一致）
            posture = 1.0            # 姿态奖励（与 flat 一致）
    
    class commands(SiriusFlatCfg.commands):
        # 🎯 课程学习：命令速度也从简单开始逐渐增加
        curriculum = True
        max_curriculum = 0.6  # 最大前进速度命令（m/s）
        max_reverse_curriculum = 0.15  # 🔧 最大后退速度命令（m/s）- 限制后退速度以保证安全
        
        class ranges:
            lin_vel_x = [-0.1, 0.3]     # 🎯 初始前进速度范围较小（从0.5而非0.8开始）
            lin_vel_y = [-0.1, 0.1]   # 🎯 初始横向速度范围较小
            ang_vel_yaw = [-0.6, 0.6]   # 🔧 降低转向速度（从 ±0.8 改为 ±0.6，约 ±34°/s）
            heading = [-3.14, 3.14]

    class camera(SiriusFlatCfg.camera):
        """相机配置 - 与 sirius 保持一致 + 渐进式数据增强提升泛化能力"""
        enable = True
        width = 87
        height = 58
        horizontal_fov = 90.0
        max_depth = 5.0
        near_plane = 0.05
        far_plane = 10.0
        enable_tensors = True  # GPU 优化
        use_collision_geometry = False
        
        # 🔍 启用调试输出：定期保存深度图到磁盘
        debug_outputs = False
        display_interval_steps = 50  # 每50个仿真步保存一次（避免IO过载）
        
        # 🎯 渐进式数据增强（Data Augmentation Curriculum）
        # 随着训练进展，逐步增加噪声和干扰，提高对真实世界传感器缺陷的鲁棒性
        augmentation_curriculum = True  # 启用数据增强课程
        
        # 高斯噪声（模拟传感器噪声）
        noise_curriculum = True
        noise_std_range = [0.0, 0.05]  # 噪声标准差范围：从0逐步增加到0.05（5%）
        
        # 随机Dropout（模拟像素失效/遮挡）
        dropout_curriculum = True
        dropout_prob_range = [0.0, 0.15]  # Dropout概率范围：从0%逐步增加到15%
        
        # 随机条带（模拟滚动快门效应、扫描线故障）
        stripe_curriculum = True
        stripe_prob_range = [0.0, 0.1]   # 条带出现概率：从0%逐步增加到10%
        stripe_width_range = [1, 3]      # 条带宽度：1-3像素
        stripe_orientation = 'both'       # 'horizontal', 'vertical', 'both'
        
        # 椒盐噪声（模拟传感器坏点）
        salt_pepper_curriculum = True
        salt_pepper_prob_range = [0.0, 0.08]  # 椒盐噪声概率：从0%逐步增加到8%
        
        # 深度量化噪声（模拟有限精度的深度传感器）
        quantization_curriculum = True
        quantization_levels_range = [256, 64]  # 量化级别：从256级（高精度）降到64级（低精度）
        
        # 模糊效果（模拟镜头失焦、运动模糊）
        blur_curriculum = False
        blur_kernel_range = [0, 3]  # 模糊核大小：从0（无模糊）到3x3
        
        # 随机亮度/对比度调整（模拟光照变化）
        brightness_curriculum = False
        brightness_range = [0.8, 1.2]    # 亮度倍数范围：0.8-1.2x
        contrast_range = [0.8, 1.2]      # 对比度倍数范围：0.8-1.2x


class SiriusCurriculumCfgPPO(LeggedRobotCfgPPO):
    """
    Sirius 课程学习任务的 PPO 训练配置
    
    使用与 sirius_flat 相同的网络结构，但训练策略针对课程学习优化
    """
    
    # 继承共享的 vision_encoder 配置
    class vision_encoder(SiriusSharedPPOCfg.vision_encoder):
        pass
    
    # 继承共享的 policy 配置（网络结构必须相同！）
    class policy(SiriusSharedPPOCfg.policy):
        pass
    
    # 算法配置
    class algorithm(SiriusSharedPPOCfg.algorithm):
        # 🎯 课程学习需要更多探索，尤其是在早期简单地形阶段
        entropy_coef = 0.025  # 提高熵系数，鼓励探索新策略
        
        # PPO 超参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        
        # 学习配置（1024 envs）
        num_learning_epochs = 8   # 每次更新的epoch数
        num_mini_batches = 2      # mini-batch数量
        learning_rate = 2.5e-4    # 学习率（与sirius_flat相同）
        schedule = 'adaptive'     # 自适应学习率调度
        gamma = 0.99              # 折扣因子
        lam = 0.95                # GAE lambda
        desired_kl = 0.01         # 目标KL散度
        max_grad_norm = 1.0       # 梯度裁剪
    
    class runner(SiriusSharedPPOCfg.runner):
        # 实验配置
        experiment_name = "sirius_curriculum"
        run_name = ''
        
        # 🎯 训练迭代：课程学习需要足够的迭代来完成所有难度级别
        max_iterations = 3000  # 增加到3000次迭代，确保有足够时间完成课程
        
        # 数据收集
        num_steps_per_env = 24  # 每个env收集的步数
        
        # 保存和日志
        save_interval = 50
        
        # 课程学习相关
        # resume = False  # 从头开始训练课程（推荐）
        # 或者从 sirius_flat 继续：resume=True, load_run="..."
        
        # 策略类
        policy_class_name = 'VisionProprioceptionActorCritic'


class SiriusCurriculum(SiriusJoyFlat):
    """
    Sirius 课程学习环境
    
    基于 SiriusJoyFlat，使用官方 terrain.py 的 curriculum 模式
    并添加渐进式相机数据增强以提高泛化能力
    """
    
    def __init__(self, cfg: SiriusCurriculumCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # 初始化相机增强课程参数
        if hasattr(self.cfg, 'camera') and self.cfg.camera.enable:
            self._init_camera_augmentation_curriculum()
    
    def _init_camera_augmentation_curriculum(self):
        """初始化相机数据增强的课程学习参数"""
        # 当前课程进度 (0.0 到 1.0)
        self.camera_aug_progress = 0.0
        
        # 存储各种增强的当前参数
        self.current_noise_std = 0.0
        self.current_dropout_prob = 0.0
        self.current_stripe_prob = 0.0
        self.current_salt_pepper_prob = 0.0
        self.current_quantization_levels = 256
        self.current_blur_kernel = 0
        
        print("[Camera Augmentation Curriculum] Initialized with progressive augmentation")
        print(f"  - Noise: {self.cfg.camera.noise_std_range}")
        print(f"  - Dropout: {self.cfg.camera.dropout_prob_range}")
        print(f"  - Stripes: {self.cfg.camera.stripe_prob_range}")
        print(f"  - Salt&Pepper: {self.cfg.camera.salt_pepper_prob_range}")
        print(f"  - Quantization: {self.cfg.camera.quantization_levels_range}")
        print(f"  - Blur: {self.cfg.camera.blur_kernel_range}")
    
    def _update_camera_augmentation_curriculum(self):
        """
        根据地形课程进度更新相机增强参数
        
        策略：相机增强跟随地形难度同步增加
        - 地形难度低时（0.0-0.3）：几乎无增强，让策略先学会基本行走
        - 地形难度中等时（0.3-0.6）：逐步引入轻度增强
        - 地形难度高时（0.6-1.0）：应用全部增强，最大化泛化能力
        """
        if not hasattr(self.cfg.camera, 'augmentation_curriculum') or not self.cfg.camera.augmentation_curriculum:
            return
        
        # 计算平均地形难度作为课程进度指标
        if hasattr(self, 'terrain_levels'):
            avg_terrain_level = float(self.terrain_levels.float().mean())
            max_terrain_level = float(self.cfg.terrain.num_rows - 1)
            self.camera_aug_progress = min(avg_terrain_level / max_terrain_level, 1.0) if max_terrain_level > 0 else 0.0
        else:
            # 如果没有地形课程，使用训练迭代数作为进度
            self.camera_aug_progress = min(float(self.common_step_counter) / 1e6, 1.0)
        
        # 线性插值各增强参数
        progress = self.camera_aug_progress
        
        if self.cfg.camera.noise_curriculum:
            r = self.cfg.camera.noise_std_range
            self.current_noise_std = r[0] + (r[1] - r[0]) * progress
        
        if self.cfg.camera.dropout_curriculum:
            r = self.cfg.camera.dropout_prob_range
            self.current_dropout_prob = r[0] + (r[1] - r[0]) * progress
        
        if self.cfg.camera.stripe_curriculum:
            r = self.cfg.camera.stripe_prob_range
            self.current_stripe_prob = r[0] + (r[1] - r[0]) * progress
        
        if self.cfg.camera.salt_pepper_curriculum:
            r = self.cfg.camera.salt_pepper_prob_range
            self.current_salt_pepper_prob = r[0] + (r[1] - r[0]) * progress
        
        if self.cfg.camera.quantization_curriculum:
            r = self.cfg.camera.quantization_levels_range
            # 注意：量化级别是反向的（从高到低）
            self.current_quantization_levels = int(r[0] + (r[1] - r[0]) * progress)
        
        if self.cfg.camera.blur_curriculum:
            r = self.cfg.camera.blur_kernel_range
            self.current_blur_kernel = int(r[0] + (r[1] - r[0]) * progress)
    
    def _apply_camera_augmentation(self, depth_images: torch.Tensor) -> torch.Tensor:
        """
        对深度图应用数据增强
        
        Args:
            depth_images: (num_envs, H, W) 原始深度图，范围 [0, max_depth]
        
        Returns:
            augmented: (num_envs, H, W) 增强后的深度图
        """
        if not hasattr(self.cfg.camera, 'augmentation_curriculum') or not self.cfg.camera.augmentation_curriculum:
            return depth_images
        
        augmented = depth_images.clone()
        
        # 1. 高斯噪声
        if self.cfg.camera.noise_curriculum and self.current_noise_std > 0:
            noise = torch.randn_like(augmented) * self.current_noise_std * self.cfg.camera.max_depth
            augmented = augmented + noise
        
        # 2. 随机Dropout（像素遮挡）
        if self.cfg.camera.dropout_curriculum and self.current_dropout_prob > 0:
            dropout_mask = torch.rand_like(augmented) > self.current_dropout_prob
            augmented = augmented * dropout_mask.float()
        
        # 3. 随机条带
        if self.cfg.camera.stripe_curriculum and self.current_stripe_prob > 0:
            augmented = self._apply_stripes(augmented)
        
        # 4. 椒盐噪声
        if self.cfg.camera.salt_pepper_curriculum and self.current_salt_pepper_prob > 0:
            augmented = self._apply_salt_pepper_noise(augmented)
        
        # 5. 深度量化
        if self.cfg.camera.quantization_curriculum and self.current_quantization_levels < 256:
            augmented = self._apply_quantization(augmented)
        
        # 6. 模糊（需要卷积，较慢）
        if self.cfg.camera.blur_curriculum and self.current_blur_kernel > 0:
            augmented = self._apply_blur(augmented)
        
        # 7. 亮度/对比度调整
        if self.cfg.camera.brightness_curriculum:
            augmented = self._apply_brightness_contrast(augmented)
        
        # Clamp到有效范围
        augmented = torch.clamp(augmented, 0.0, self.cfg.camera.max_depth)
        
        return augmented
    
    def _apply_stripes(self, depth: torch.Tensor) -> torch.Tensor:
        """应用随机条带效果"""
        if torch.rand(1).item() > self.current_stripe_prob:
            return depth
        
        num_envs, H, W = depth.shape
        orientation = self.cfg.camera.stripe_orientation
        
        # 随机选择方向
        if orientation == 'both':
            is_horizontal = torch.rand(1).item() > 0.5
        else:
            is_horizontal = (orientation == 'horizontal')
        
        # 随机条带宽度
        width = torch.randint(
            self.cfg.camera.stripe_width_range[0],
            self.cfg.camera.stripe_width_range[1] + 1,
            (1,)
        ).item()
        
        # 随机选择一个环境应用条带
        env_idx = torch.randint(0, num_envs, (1,)).item()
        
        if is_horizontal:
            # 水平条带
            start_row = torch.randint(0, max(1, H - width), (1,)).item()
            depth[env_idx, start_row:start_row+width, :] = 0.0
        else:
            # 垂直条带
            start_col = torch.randint(0, max(1, W - width), (1,)).item()
            depth[env_idx, :, start_col:start_col+width] = 0.0
        
        return depth
    
    def _apply_salt_pepper_noise(self, depth: torch.Tensor) -> torch.Tensor:
        """应用椒盐噪声"""
        noise_mask = torch.rand_like(depth) < self.current_salt_pepper_prob
        
        # 50%概率设为0（椒），50%概率设为max_depth（盐）
        salt_mask = torch.rand_like(depth) > 0.5
        noise_values = torch.where(salt_mask, 
                                   torch.full_like(depth, self.cfg.camera.max_depth),
                                   torch.zeros_like(depth))
        
        depth = torch.where(noise_mask, noise_values, depth)
        return depth
    
    def _apply_quantization(self, depth: torch.Tensor) -> torch.Tensor:
        """应用深度量化（降低精度）"""
        levels = self.current_quantization_levels
        max_depth = self.cfg.camera.max_depth
        
        # 量化到指定级别
        quantized = torch.round(depth / max_depth * levels) / levels * max_depth
        return quantized
    
    def _apply_blur(self, depth: torch.Tensor) -> torch.Tensor:
        """应用高斯模糊（简化版：使用平均池化）"""
        if self.current_blur_kernel <= 0:
            return depth
        
        # 使用平均池化模拟模糊（更高效）
        kernel_size = self.current_blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保是奇数
        
        # 添加batch和channel维度
        depth_4d = depth.unsqueeze(1)  # (N, 1, H, W)
        
        # 使用avg_pool2d模拟模糊
        padding = kernel_size // 2
        blurred = torch.nn.functional.avg_pool2d(
            depth_4d, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding
        )
        
        return blurred.squeeze(1)  # (N, H, W)
    
    def _apply_brightness_contrast(self, depth: torch.Tensor) -> torch.Tensor:
        """应用随机亮度和对比度调整"""
        if not self.cfg.camera.brightness_curriculum:
            return depth
        
        # 随机亮度倍数
        brightness_factor = torch.empty(1).uniform_(
            self.cfg.camera.brightness_range[0],
            self.cfg.camera.brightness_range[1]
        ).item()
        
        # 随机对比度倍数
        contrast_factor = torch.empty(1).uniform_(
            self.cfg.camera.contrast_range[0],
            self.cfg.camera.contrast_range[1]
        ).item()
        
        # 应用亮度和对比度
        # contrast: (depth - mean) * factor + mean
        # brightness: result * factor
        mean_depth = depth.mean()
        depth = (depth - mean_depth) * contrast_factor + mean_depth
        depth = depth * brightness_factor
        
        return depth
    
    def compute_observations(self):
        """重写观测计算，添加相机增强"""
        # 先更新相机增强课程
        self._update_camera_augmentation_curriculum()
        
        # 调用父类方法获取原始观测
        super().compute_observations()
        
        # 如果启用了相机增强，应用增强到深度观测
        if hasattr(self, 'depth_obs_buf') and hasattr(self.cfg.camera, 'augmentation_curriculum'):
            if self.cfg.camera.augmentation_curriculum and self.depth_obs_buf is not None:
                # depth_obs_buf shape: (num_envs, 1, H, W)
                # 移除channel维度进行增强
                depth_2d = self.depth_obs_buf.squeeze(1) * self.cfg.camera.max_depth  # 反归一化
                
                # 应用增强
                depth_augmented = self._apply_camera_augmentation(depth_2d)
                
                # 重新归一化并添加channel维度
                self.depth_obs_buf = (depth_augmented / self.cfg.camera.max_depth).unsqueeze(1)
    
    def post_physics_step(self):
        """重写物理步进后的处理，添加增强日志"""
        super().post_physics_step()
        
        # 定期记录增强参数
        if self.common_step_counter % 1000 == 0 and hasattr(self, 'camera_aug_progress'):
            if self.cfg.camera.augmentation_curriculum:
                print(f"\n[Camera Aug Progress] {self.camera_aug_progress:.3f}")
                print(f"  Noise std: {self.current_noise_std:.4f}")
                print(f"  Dropout prob: {self.current_dropout_prob:.4f}")
                print(f"  Stripe prob: {self.current_stripe_prob:.4f}")
                print(f"  Salt&Pepper prob: {self.current_salt_pepper_prob:.4f}")
                print(f"  Quantization levels: {self.current_quantization_levels}")
                print(f"  Blur kernel: {self.current_blur_kernel}")
    
    def _reset_root_states(self, env_ids):
        """
        重置机器人根状态（位置和速度）
        
        🎯 覆盖父类方法，使用正常的课程学习生成逻辑（不带桥梁偏移）
        
        与父类 SiriusJoyFlat._reset_root_states 的区别：
        - ❌ 移除桥梁偏移（bridge_offset_x = -4）
        - ✅ 使用标准的地形中心生成
        - ✅ 在小范围内随机生成（0.5m 半径）
        
        Args:
            env_ids (Tensor[int64]): 需要重置的环境ID
        """
        if len(env_ids) == 0:
            return
        
        # 确保是 long 类型
        env_ids = env_ids.to(dtype=torch.long)
        
        # ------------------- 基础位置设置 -------------------
        # 1. 从初始状态开始
        self.root_states[env_ids] = self.base_init_state
        
        # 2. 添加环境原点偏移（地形中心）
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # 3. 在地形中心附近随机一个位置（半径 0.5m 的圆内）
        #    使用极坐标采样确保均匀分布
        R = 0.5  # 半径 (m)
        r = R * torch.sqrt(torch.rand(len(env_ids), 1, device=self.device))
        theta = 2.0 * 3.14159265359 * torch.rand(len(env_ids), 1, device=self.device)
        offset_xy = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        self.root_states[env_ids, :2] += offset_xy
        
        # ------------------- 基础速度设置 -------------------
        # 随机初始速度 [-0.5, 0.5] m/s 或 rad/s
        # [7:10]: 线速度 (x, y, z)
        # [10:13]: 角速度 (roll, pitch, yaw)
        self.root_states[env_ids, 7:13] = torch.rand(len(env_ids), 6, device=self.device) * 1.0 - 0.5
        
        # ------------------- 更新到仿真 -------------------
        # 计算 actor ID（每个 env 对应的 actor）
        actor_ids = (env_ids * int(self.actors_per_env)).to(dtype=torch.long)
        actor_ids_int32 = actor_ids.to(dtype=torch.int32)
        
        # 更新本地 actor buffer
        self._actor_root_states[actor_ids] = self.root_states[env_ids]
        
        # 同步到 Isaac Gym 仿真
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._actor_root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )
