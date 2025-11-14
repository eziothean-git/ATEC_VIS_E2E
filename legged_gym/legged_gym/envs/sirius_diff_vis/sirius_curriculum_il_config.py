# sirius_curriculum_il_config.py
"""
Sirius Curriculum IL 训练配置

🎯 目标：通过模仿学习(IL)将无视觉的教师策略迁移到有视觉的学生策略

训练策略：
- 阶段1：在平地上进行 RL + IL 并行训练
  - 教师：冻结的本体感知策略（来自 task=sirius）
  - 学生：视觉-本体融合策略
  - 损失：RL奖励 + IL蒸馏损失（动作匹配）
  
- 阶段2：在 curriculum 地形上纯 RL 微调
  - 仅使用 RL 奖励，无 IL 损失
  - 从中等难度地形开始
  - 见 sirius_curriculum_finetune_config.py

使用方法（阶段1）：
    python train.py --task=sirius_curriculum_il --num_envs=1024 --headless \\
        --teacher_model=logs/sirius/YYYY-MM-DD/HH-MM-SS/model_1800.pt
"""

from .sirius_curriculum_config import (
    SiriusCurriculumCfg, 
    SiriusCurriculumCfgPPO, 
    SiriusCurriculum
)
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class SiriusCurriculumILCfg(SiriusCurriculumCfg):
    """
    IL训练的环境配置 - 阶段1: 平地并行训练
    
    特点：
    - 地形：平地（与教师策略训练环境一致）
    - 相机：启用，但数据增强较保守
    - 命令：与教师策略相同的速度范围
    """
    
    class env(SiriusCurriculumCfg.env):
        num_envs = 1024  # IL训练可以用较少的env（教师指导更高效）
        episode_length_s = 20  # 适中的episode长度
    
    class terrain(SiriusCurriculumCfg.terrain):
        # 🔧 修改：使用 curriculum 地形（与教师策略训练环境一致）
        mesh_type = "trimesh"  # Curriculum 地形
        curriculum = True       # 启用地形课程
        measure_heights = True  # 需要高度测量
        
        # 从较低难度开始
        max_init_terrain_level = 1
        
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 8
        
        curriculum_distribution = {
            'easy_ratio': 0.80,
            'medium_ratio': 0.15,
            'exploration_ratio': 0.05,
        }
        
        skip_terrain_types = [4, 5]  # 跳过障碍物、踏脚石
        
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 20.0
        terrain_proportions = [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        
    class camera(SiriusCurriculumCfg.camera):
        """相机配置 - 保守的数据增强策略"""
        enable = True  # 必须启用相机
        
        # 相机参数与curriculum保持一致
        width = 87
        height = 58
        horizontal_fov = 90.0
        max_depth = 5.0
        
        # 🎯 数据增强课程：初期更保守
        augmentation_curriculum = True
        
        # 减少噪声范围（初期让视觉编码器更容易学习）
        noise_curriculum = True
        noise_std_range = [0.0, 0.02]  # 0-2% (vs 0-5% in curriculum)
        
        dropout_curriculum = True
        dropout_prob_range = [0.0, 0.05]  # 0-5% (vs 0-15% in curriculum)
        
        stripe_curriculum = True
        stripe_prob_range = [0.0, 0.05]  # 0-5% (vs 0-10% in curriculum)
        
        salt_pepper_curriculum = True
        salt_pepper_prob_range = [0.0, 0.04]  # 0-4% (vs 0-8% in curriculum)
        
        quantization_curriculum = True
        quantization_levels_range = [256, 128]  # 128级 (vs 64级 in curriculum)
        
        blur_curriculum = True
        blur_kernel_range = [0, 2]  # 最大2x2 (vs 3x3 in curriculum)
        
        brightness_curriculum = True
        brightness_range = [0.9, 1.1]  # ±10% (vs ±20% in curriculum)
        contrast_range = [0.9, 1.1]
        
        # 调试输出
        debug_outputs = True
        display_interval_steps = 24000  # 每24000步显示一次（与训练步数匹配）
        
    class commands(SiriusCurriculumCfg.commands):
        """命令配置 - 与教师策略保持一致（使用 heading_command 和课程）"""
        # 🔧 使用命令课程（与教师策略相同）
        curriculum = True
        max_curriculum = 0.8
        max_reverse_curriculum = 0.1
        min_forward_speed = 0.2
        curriculum_step = 0.1
        curriculum_threshold = 0.7
        
        # 🔥 关键：使用 heading_command 模式，朝向与速度方向对齐
        heading_command = True
        
        class ranges:
            # 与教师策略相同的范围
            lin_vel_x = [-0.1, 0.3]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]
            
    class rewards(SiriusCurriculumCfg.rewards):
        """奖励配置 - 与教师策略保持一致"""
        # 使用与 sirius_flat 类似的奖励权重
        only_positive_rewards = True  # 简化奖励结构
        
        class scales:
            # 主要奖励：速度跟踪
            tracking_lin_vel = 15
            tracking_ang_vel = 10
            
            # 姿态控制
            orientation = -0.5
            base_height = -0.5
            
            # 平滑性
            lin_vel_z = -0.0125
            ang_vel_xy = -0.05
            action_rate = -0.5
            
            # 姿态保持
            posture = 0.5
            
            # 碰撞和滑移
            collision = -2.5
            slip = -0.5
            
            # 站立静止
            stand_still = -2
            
            # 足部控制
            feet_air_time = 0.5
            stumble = -0.5
            feet_contact_number = -5
            
            # 终止惩罚
            termination = -1000.0


class SiriusCurriculumILCfgPPO(SiriusCurriculumCfgPPO):
    """
    IL训练的PPO配置 - 阶段1
    
    关键特性：
    - 启用 IL 损失（动作匹配）
    - 较高学习率（教师指导加速学习）
    - 较低熵系数（减少随机探索，更多模仿）
    """
    
    # 继承共享的 vision_encoder 配置
    class vision_encoder(SiriusCurriculumCfgPPO.vision_encoder):
        pass
    
    # 继承共享的 policy 配置
    class policy(SiriusCurriculumCfgPPO.policy):
        pass
    
    class algorithm(SiriusCurriculumCfgPPO.algorithm):
        """算法配置 - IL训练特化"""
        
        # ========== IL 相关配置 ==========
        use_imitation_loss = True  # 启用模仿学习损失
        imitation_loss_coef = 1.0  # IL损失权重（与RL损失平衡）
        
        # 可选：逐步降低IL权重（curriculum on IL loss）
        imitation_curriculum = True
        imitation_coef_schedule = {
            'start': 1.0,    # 初始权重
            'end': 0.025,      # 最终权重
            'iterations': 1250,  # 在1250次迭代内线性衰减
        }
        
        # ========== PPO 超参数 ==========
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        
        # 学习率：比纯RL稍高（教师指导加速学习）
        learning_rate = 5e-4  # vs 2.5e-4 in curriculum
        schedule = 'adaptive'
        
        # 熵系数：降低随机探索（更多模仿教师）
        entropy_coef = 0.0075  # vs 0.015 in curriculum
        
        # 梯度和KL散度
        desired_kl = 0.01  # 更保守（避免偏离教师太远）
        max_grad_norm = 1.0
        
        # 折扣因子
        gamma = 0.99
        lam = 0.95
        
        # 学习epochs
        num_learning_epochs = 8  # 增加epochs利用教师信号
        num_mini_batches = 4
    
    class runner(SiriusCurriculumCfgPPO.runner):
        """训练运行配置"""
        
        # 实验名称
        experiment_name = "sirius_curriculum_il"
        run_name = 'stage1_curriculum_parallel'
        
        # ========== 教师模型配置 ==========
        # 🔥 设置教师模型路径
        teacher_model_path = "/home/eziothean/ATEC_VIS_E2E/legged_gym/logs/sirius_teacher_curriculum/Nov14_22-58-38_/model_2250.pt"
        
        # ========== 训练配置 ==========
        max_iterations = 3500  # 阶段1训练迭代数
        num_steps_per_env = 48
        
        # 保存和日志
        save_interval = 50  # 每50次迭代保存一次
        
        # 策略类（与curriculum保持一致）
        policy_class_name = 'VisionProprioceptionActorCritic'
        
        # 🎓 算法类（使用PPOIL而不是PPO）
        algorithm_class_name = 'PPOIL'
        
        # 从头开始训练（不从checkpoint恢复）
        resume = False
        load_run = -1
        checkpoint = -1


class SiriusCurriculumIL(SiriusCurriculum):
    """
    IL训练环境 - 继承自 SiriusCurriculum
    
    主要特点：
    - 使用 curriculum 地形（与教师策略一致）
    - 使用 heading_command 模式（朝向与速度对齐）
    - 相机增强、命令课程等与教师策略保持一致
    """
    
    def __init__(self, cfg: SiriusCurriculumILCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        print("\n" + "="*80)
        print("🔥 [SiriusCurriculumIL] IL Training Environment initialized")
        print(f"  - Terrain: {self.cfg.terrain.mesh_type} (curriculum={self.cfg.terrain.curriculum})")
        print(f"  - Camera: {self.cfg.camera.enable}")
        print(f"  - Num envs: {self.cfg.env.num_envs}")
        print(f"  - heading_command: {self.cfg.commands.heading_command}")
        print(f"  - Command curriculum: {self.cfg.commands.curriculum}")
        print("="*80 + "\n")
    
    # 所有其他功能继承自 SiriusCurriculum，包括：
    #   - _post_physics_step_callback（支持 heading_command）
    #   - _resample_commands（朝向对齐采样）
    #   - 相机数据增强
    #   - 地形课程
    #   - 命令课程
    #   - 奖励计算
    #   - 重置逻辑
