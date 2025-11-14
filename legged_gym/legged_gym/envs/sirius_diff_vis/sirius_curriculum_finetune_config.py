# sirius_curriculum_finetune_config.py
"""
Sirius Curriculum 后训练/微调配置

🎯 目标：在复杂地形上微调已在平地上通过IL训练的视觉策略

训练策略：
- 从阶段1（平地IL训练）的checkpoint继续
- 在curriculum地形上进行纯RL训练（无IL损失）
- 从中等难度地形开始（跳过最简单的）
- 启用完整的相机数据增强

使用方法：
    python train.py --task=sirius_curriculum_finetune --num_envs=512 --headless \\
        --resume --load_run=logs/sirius_curriculum_il/stage1_flat_parallel
"""

from .sirius_curriculum_config import (
    SiriusCurriculumCfg,
    SiriusCurriculumCfgPPO,
    SiriusCurriculum
)


class SiriusCurriculumFinetuneCfg(SiriusCurriculumCfg):
    """
    后训练/微调的环境配置
    
    特点：
    - 地形：完整的curriculum（多种地形类型和难度）
    - 相机：完整的数据增强
    - 起始难度：中等（跳过最简单的地形）
    """
    
    class env(SiriusCurriculumCfg.env):
        num_envs = 512  # 微调阶段可以用较少env（已有基础策略）
        episode_length_s = 24  # 保持与curriculum一致
    
    class terrain(SiriusCurriculumCfg.terrain):
        """完整的地形课程配置"""
        mesh_type = "trimesh"  # 使用heightfield地形
        
        # 启用课程学习
        curriculum = True
        selected = False
        measure_heights = True
        
        # 地形布局
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # 10个难度级别
        num_cols = 8   # 8种地形类型
        
        # 🔧 起始难度：从中等难度开始（因为已有平地基础）
        max_init_terrain_level = 3  # 难度3-4（中等斜坡、台阶）
        
        # 地形分布策略（与curriculum一致）
        curriculum_distribution = {
            'easy_ratio': 0.80,
            'medium_ratio': 0.15,
            'exploration_ratio': 0.05,
        }
        
        # 跳过难以学习的地形类型
        skip_terrain_types = [4, 5]  # 障碍物、踏脚石
        
        # 地形参数
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 20.0
        
        # 地形类型比例（与curriculum一致）
        terrain_proportions = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
        
        # 摩擦系数
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
    
    class camera(SiriusCurriculumCfg.camera):
        """完整的相机数据增强"""
        enable = True
        
        width = 87
        height = 58
        horizontal_fov = 90.0
        max_depth = 5.0
        
        # 🎯 启用完整数据增强（提高泛化能力）
        augmentation_curriculum = True
        
        # 完整噪声范围
        noise_curriculum = True
        noise_std_range = [0.0, 0.05]  # 0-5%
        
        dropout_curriculum = True
        dropout_prob_range = [0.0, 0.15]  # 0-15%
        
        stripe_curriculum = True
        stripe_prob_range = [0.0, 0.1]  # 0-10%
        stripe_width_range = [1, 3]
        stripe_orientation = 'both'
        
        salt_pepper_curriculum = True
        salt_pepper_prob_range = [0.0, 0.08]  # 0-8%
        
        quantization_curriculum = True
        quantization_levels_range = [256, 64]  # 256 → 64级
        
        blur_curriculum = True
        blur_kernel_range = [0, 3]  # 0 → 3x3
        
        brightness_curriculum = True
        brightness_range = [0.8, 1.2]  # ±20%
        contrast_range = [0.8, 1.2]
        
        # 调试输出
        debug_outputs = True
        display_interval_steps = 2000
        
        # 性能优化
        update_interval = 2  # 每2步更新一次相机
    
    class commands(SiriusCurriculumCfg.commands):
        """命令课程 - 恢复启用"""
        curriculum = True
        max_curriculum = 0.8
        max_reverse_curriculum = 0.1
        min_forward_speed = 0.2
        curriculum_step = 0.1
        curriculum_threshold = 0.5
        
        class ranges:
            # 初始范围（会通过curriculum扩展）
            lin_vel_x = [-0.1, 0.3]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]
    
    class rewards(SiriusCurriculumCfg.rewards):
        """奖励配置 - 与curriculum保持一致"""
        only_positive_rewards = False  # 允许负奖励（更精细的反馈）
        
        class scales:
            tracking_lin_vel = 15
            tracking_ang_vel = 10
            orientation = -0.5
            base_height = -0.5
            lin_vel_z = -0.0125
            ang_vel_xy = -0.05
            action_rate = -0.5
            posture = 0.5
            collision = -2.5
            slip = -0.5
            stand_still = -2
            feet_air_time = 0.5
            stumble = -0.5
            feet_contact_number = -5
            termination = -1000.0


class SiriusCurriculumFinetuneCfgPPO(SiriusCurriculumCfgPPO):
    """
    后训练/微调的PPO配置
    
    关键特性：
    - 关闭IL损失（纯RL训练）
    - 较低学习率（微调阶段）
    - 恢复正常熵系数（鼓励探索新地形）
    """
    
    # 继承共享的 vision_encoder 配置
    class vision_encoder(SiriusCurriculumCfgPPO.vision_encoder):
        pass
    
    # 继承共享的 policy 配置
    class policy(SiriusCurriculumCfgPPO.policy):
        pass
    
    class algorithm(SiriusCurriculumCfgPPO.algorithm):
        """算法配置 - 纯RL微调"""
        
        # ========== 关闭 IL ==========
        use_imitation_loss = False  # 不使用教师指导
        
        # ========== PPO 超参数 ==========
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        
        # 学习率：降低（微调阶段，避免破坏已学到的知识）
        learning_rate = 1e-4  # 从5e-4降到1e-4
        schedule = 'adaptive'
        
        # 熵系数：恢复正常（鼓励探索新地形）
        entropy_coef = 0.015  # 恢复到curriculum水平
        
        # 梯度和KL散度
        desired_kl = 0.015  # 允许更大的策略更新
        max_grad_norm = 1.0
        
        # 折扣因子
        gamma = 0.99
        lam = 0.95
        
        # 学习epochs
        num_learning_epochs = 10  # 保持与curriculum一致
        num_mini_batches = 4
    
    class runner(SiriusCurriculumCfgPPO.runner):
        """训练运行配置 - 从阶段1继续"""
        
        # 实验名称
        experiment_name = "sirius_curriculum_il"  # 保持相同experiment（方便对比）
        run_name = 'stage2_curriculum_finetune'
        
        # ========== 从阶段1继续 ==========
        resume = True  # 必须启用
        # load_run 和 checkpoint 需要在命令行指定
        # 例如: --load_run=logs/sirius_curriculum_il/stage1_flat_parallel --checkpoint=-1
        
        # ========== 训练配置 ==========
        max_iterations = 1500  # 阶段2微调迭代数
        num_steps_per_env = 32
        
        # 保存和日志
        save_interval = 50
        
        # 策略类
        policy_class_name = 'VisionProprioceptionActorCritic'


class SiriusCurriculumFinetune(SiriusCurriculum):
    """
    后训练/微调环境 - 继承自 SiriusCurriculum
    
    功能：
    - 完整的curriculum地形
    - 完整的相机数据增强
    - 从中等难度开始
    """
    
    def __init__(self, cfg: SiriusCurriculumFinetuneCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        print("[Fine-tuning] Environment initialized")
        print(f"  - Terrain: {self.cfg.terrain.mesh_type} (curriculum={self.cfg.terrain.curriculum})")
        print(f"  - Starting difficulty: {self.cfg.terrain.max_init_terrain_level}")
        print(f"  - Camera augmentation: {self.cfg.camera.augmentation_curriculum}")
        print(f"  - Num envs: {self.cfg.env.num_envs}")
        print(f"  - Skipped terrain types: {self.cfg.terrain.skip_terrain_types}")
    
    # 所有功能继承自 SiriusCurriculum
    # 包括：
    #   - 地形课程更新
    #   - 命令课程更新
    #   - 相机数据增强
    #   - 奖励计算
    #   - 等等
