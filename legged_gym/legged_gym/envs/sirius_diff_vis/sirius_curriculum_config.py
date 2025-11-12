# sirius_curriculum_config.py
"""
Sirius 课程学习配置 - 使用官方 terrain.py 的 curriculum 模式训练泛化能力

课程学习布局：
- Rows (难度): 从易到难 (0.0 → 1.0)
- Cols (地形类型): 不同障碍类型
  - Col 0-1: 斜坡 (上坡/下坡)
  - Col 2: 斜坡 + 粗糙表面
  - Col 3-4: 台阶 (上升/下降)
  - Col 5: 离散障碍物
  - Col 6: 踏脚石
  - Col 7: 缝隙
  - Col 8: 坑洞

训练策略：
1. 机器人从简单地形（第一行）开始训练
2. 成功后自动晋级到更难的地形（更高行数）
3. 最终在所有地形类型和难度级别上都能泛化

使用方法：
    python train.py --task=sirius_curriculum --num_envs=1024 --headless
"""

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
        num_observations = 45
        episode_length_s = 100.0  # 稍短一些，加快curriculum迭代
        num_envs = 1024  # 建议用较多envs覆盖更多地形

    class terrain(SiriusFlatCfg.terrain):
        mesh_type = "trimesh"  # 必须用 trimesh 或 heightfield
        
        # ⚠️ 启用课程学习！
        curriculum = True
        selected = False
        
        # 是否测量高度（可选，增加观测维度）
        measure_heights = False  # 先关闭，专注视觉
        
        # 地形网格布局
        terrain_length = 8.0  # 每个子地形的长度（米）
        terrain_width = 8.0   # 每个子地形的宽度（米）
        num_rows = 5          # 难度级别数量（5个难度：0.0, 0.25, 0.5, 0.75, 1.0）
        num_cols = 8          # 地形类型数量（8种地形）- 修正：make_terrain实际只有8种
        
        # 🔥 关键修复：初始地形难度范围必须 < num_rows
        max_init_terrain_level = 4  # 最大初始难度级别（索引0-4，对应num_rows=5）
        
        # 地形分辨率
        horizontal_scale = 0.1  # 0.1m per pixel
        vertical_scale = 0.005  # 5mm per unit
        border_size = 20.0      # 边界大小（米）
        
        # 地形类型比例（对应 make_terrain 中的8种类型）
        # [斜坡, 斜坡+粗糙, 台阶上, 台阶下, 障碍物, 踏脚石, 缝隙, 坑洞]
        terrain_proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
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
            orientation = -0.5          # 惩罚倾斜
            base_height = -0.0          # 保持高度（可选）
            
            # 平滑性奖励
            lin_vel_z = -0.5            # 惩罚垂直速度
            ang_vel_xy = -0.05          # 惩罚横滚/俯仰角速度
            dof_acc = -2.5e-7           # 惩罚关节加速度
            collision = -1.0            # 惩罚碰撞
            action_rate = -0.01         # 惩罚动作变化率
            
            # 能量效率
            torques = -0.00001          # 惩罚大扭矩
            dof_vel = -0.0              # 惩罚大关节速度
            
            # 步态奖励
            feet_air_time = 1.0         # 奖励腾空时间
            stumble = -2.0              # 惩罚绊倒（对崎岖地形重要）
            stand_still = -0.25          # 惩罚原地不动
            
            # 终止惩罚
            termination = -0.0

            orientation = -5.0       # 强姿态惩罚（与 flat 一致）
            feet_air_time = 3.0      # 腾空奖励（与 flat 一致）
            base_height = -200.0     # 强高度惩罚（与 flat 一致）
            posture = 1.0            # 姿态奖励（与 flat 一致）
    
    class commands(SiriusFlatCfg.commands):
        # 课程学习：命令速度范围也会逐渐增加
        curriculum = True
        max_curriculum = 0.8  # 最大线速度命令（m/s）
        
        class ranges:
            lin_vel_x = [-0.15, 0.8]    # 前进速度范围
            lin_vel_y = [-0.2, 0.2]    # 横向速度范围
            ang_vel_yaw = [-1.0, 1.0]  # 转向速度范围
            heading = [-3.14, 3.14]

    class camera(SiriusFlatCfg.camera):
        """相机配置 - 与 sirius 保持一致 + 启用调试输出"""
        enable = True
        width = 87
        height = 58
        horizontal_fov = 90.0
        max_depth = 5.0
        near_plane = 0.1
        far_plane = 10.0
        enable_tensors = True  # GPU 优化
        use_collision_geometry = False
        
        # 🔍 启用调试输出：定期保存深度图到磁盘
        debug_outputs = False
        display_interval_steps = 50  # 每50个仿真步保存一次（避免IO过载）


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
        # 课程学习可能需要更多探索
        entropy_coef = 0.01  # 熵系数，鼓励探索
        
        # PPO 超参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        
        # 学习配置（1024 envs）
        num_learning_epochs = 8   # 每次更新的epoch数
        num_mini_batches = 2      # mini-batch数量
        learning_rate = 2.5e-4    # 学习率（与sirius_flat相同）
        schedule = 'adaptive'     # 学习率调度
        gamma = 0.99              # 折扣因子
        lam = 0.95                # GAE lambda
        desired_kl = 0.01         # 目标KL散度
        max_grad_norm = 1.0       # 梯度裁剪
    
    class runner(SiriusSharedPPOCfg.runner):
        # 实验配置
        experiment_name = "sirius_curriculum"
        run_name = ''
        
        # 训练迭代
        max_iterations = 2000  # 课程学习需要更多迭代
        
        # 数据收集
        num_steps_per_env = 24  # 每个env收集的步数
        
        # 保存和日志
        save_interval = 50
        
        # 课程学习相关
        # resume = False  # 从头开始训练课程
        # 或者从 sirius_flat 继续：resume=True, load_run="..."
        
        # 策略类
        policy_class_name = 'VisionProprioceptionActorCritic'


class SiriusCurriculum(SiriusJoyFlat):
    """
    Sirius 课程学习环境
    
    基于 SiriusJoyFlat，使用官方 terrain.py 的 curriculum 模式
    """
    
    def __init__(self, cfg: SiriusCurriculumCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    # 可以在这里添加课程学习特定的方法
    # 例如：监控各难度级别的成功率，动态调整训练策略等
