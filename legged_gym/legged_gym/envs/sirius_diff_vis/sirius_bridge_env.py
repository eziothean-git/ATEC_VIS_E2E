# sirius_two_span_bridge.py

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.sirius_diff_vis.sirius_flat_config import SiriusFlatCfg
from legged_gym.envs.sirius_diff_vis.sirius_joystick import SiriusJoyFlat
from .sirius_shared_model import SiriusSharedPPOCfg


class SiriusTwoSpanBridgeCfg(SiriusFlatCfg):
    """
    两段桥面吊桥环境配置。

    坐标与尺寸约定：
    - x 轴：沿桥前进方向（机器人走的方向）
    - y 轴：垂直于桥的横向方向（桥面“宽度”方向）

    木板与平台的几何说明：
    - 每一块木板在 y 方向长度为 1.0 m
    - 每一块木板在 x 方向厚度为 0.25 m
    - 出发区 / 中间平台 / 终点平台：
        * y 方向为 1.0 m
        * x 方向为 0.75 m
    """

    class env(SiriusFlatCfg.env):
        # 观测维度必须与 sirius 任务一致！
        num_observations = 45  # 本体观测维度 (不包含视觉，视觉通过 depth_obs_buf 单独传递)
        episode_length_s = 30.0

        num_envs = 512  # 桥梁场景：减少env数量以适应更复杂的场景

    # 初始姿态/关节角沿用 SiriusFlatCfg.init_state
    # 如果你要改起始点（圆半径 0.25 m），可以在这里单独写一个 init_state 覆盖，
    # 或者在 SiriusJoyFlat._reset_root_states 里改，我下面留了接口注释。

    class terrain(SiriusFlatCfg.terrain):
        # 使用自定义三角网格地形
        mesh_type = "trimesh"
        measure_heights = False
        curriculum = False
        selected = True

        horizontal_scale = 0.01
        vertical_scale = 0.01
        border_size = 0.0

        # 整体地形尺寸，给一个略大于桥总长度和宽度的范围
        terrain_length = 12.0  # x 方向
        terrain_width = 4.0    # y 方向
        num_rows = 1
        num_cols = 1

        # ===== 两段桥的参数设定 =====
        # 每块木板：x 方向 0.25 m（你说的“宽 25 cm”），y 方向 1.0 m（“长 1 m”）
        plank_len_x = 0.25

        # 第一段：板间距 0.5 m，对应 x 方向上板与板之间的空隙
        gap1_x = 0.05
        # 第二段：板间距 0.15 m
        gap2_x = 0.15

        # 选取板子数量使总长度大致接近 3.05 m 和 4.15 m
        N1 = 10    
        N2 = 10

        # 休息平台（出发区 / 中间平台 / 终点平台）在 x 方向长度 0.75 m
        # （y 方向为 1.0 m，由 bridge_width 控制）
        rest_len_x = 0.75

        # bridge_width 表示 y 方向宽度：这里为 1.0 m，对应“长 1 m”
        # 注意：每块木板和休息平台在 y 方向都占满这 1.0 m
        terrain_kwargs = {
            "type": "bridge_terrain",
            "bridge_width": 1.0,         # y 方向 1 m
            "platform_height": 1.0,      # 顶面离地高度 1 m
            "gap_depth": 1.0,            # 缝隙向下的深度（掉下去的高度差）

            # platform_lengths 和 gap_lengths 都是沿 x 方向的尺寸
            #
            # platform_lengths:
            #   [出发区] + [第一段 N1 块木板] + [中间平台] + [第二段 N2 块木板] + [终点平台]
            "platform_lengths": (
                [rest_len_x] +                        # 出发平台
                [plank_len_x] * N1 +                  # 第一段 N1 块板
                [rest_len_x] +                        # 中间平台
                [plank_len_x] * N2 +                  # 第二段 N2 块板
                [rest_len_x]                          # 终点平台
            ),

            # gap_lengths: 相邻 platform 之间的间隙（长度 = len(platform_lengths) - 1）
            # [出发区-第一块板] + (第一段内部间隙) + [第一段末板-中间平台] +
            # [中间平台-第二段首板] + (第二段内部间隙) + [第二段末板-终点平台]
            "gap_lengths": (
                [0.0] +                               # 出发区 与 第一块板紧贴
                [gap1_x] * (N1 - 1) +                 # 第一段板间距 0.5
                [0.0] +                               # 第一段末板 与 中间平台紧贴
                [0.0] +                               # 中间平台 与 第二段首板紧贴
                [gap2_x] * (N2 - 1) +                 # 第二段板间距 0.15
                [0.0]                                 # 第二段末板 与 终点平台紧贴
            ),
        }

        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0

    # Enable camera for visualization in this diff_vis task by default
    class camera(SiriusFlatCfg.camera):
        enable = True
        # 继承 SiriusFlatCfg 的相机配置（87x58 分辨率）
        # display every 3 simulation steps (this is in sim steps, not seconds)
        display_interval_steps = 3
        # window name shown by OpenCV
        display_window_name = "sirius_diff_vis_cam"
        # max depth 继承自 SiriusFlatCfg.camera (5.0m)


class SiriusTwoSpanBridgeCfgPPO(LeggedRobotCfgPPO):
    """
    Sirius 两段桥任务的 PPO 训练配置。
    
    继承共享模型配置 (SiriusSharedPPOCfg)，确保与 sirius (平地) 任务的网络结构一致。
    这样可以使用 --resume 从平地训练的权重继续训练。
    
    只覆盖 runner 中的任务特定参数（实验名、迭代次数等）。
    """
    
    # ⚠️ 必须继承 vision_encoder 配置！
    class vision_encoder(SiriusSharedPPOCfg.vision_encoder):
        # 使用共享配置的视觉编码器（与 sirius 平地任务完全相同）
        pass
    
    # 继承共享的 policy 配置（网络结构）- 必须与 sirius 任务完全相同！
    class policy(SiriusSharedPPOCfg.policy):
        # 使用共享配置的 [256, 128, 64]
        # 不要在这里覆盖 actor_hidden_dims 或 critic_hidden_dims！
        pass
    
    # 继承共享的 algorithm 配置（PPO 超参数）
    class algorithm(SiriusSharedPPOCfg.algorithm):
        # 阶段2：512 envs，学习率相应减半（相比阶段1的1024 envs）
        # 线性缩放：lr_stage2 = lr_stage1 * (num_envs_stage2 / num_envs_stage1)
        # = 2.5e-4 * (512 / 1024) = 1.25e-4
        pass

    # 任务特定的 runner 配置
    class runner(SiriusSharedPPOCfg.runner):
        experiment_name = "sirius_bridge"  # 阶段2: 桥梁训练
        max_iterations = 1200
        
        # 学习率根据 num_envs 线性缩放
        # 阶段1 (sirius): 1024 envs, lr=2.5e-4
        # 阶段2 (sirius_diff_vis): 512 envs, lr=1.25e-4 (减半)
        learning_rate = 1.25e-4
        
        # Resume 相关配置（从阶段1继续训练时使用）
        # 使用方法：python train.py --task=sirius_diff_vis --resume
        # 会自动加载 experiment_name="sirius_flat" 的最新 checkpoint
        # 如果需要明确指定，可以用 CLI 参数覆盖：
        # --load_run="Nov11_10-30-45_sirius_flat" --checkpoint=1000


class SiriusTwoSpanBridge(SiriusJoyFlat):
    """
    基于 SiriusJoyFlat 的两段桥环境。

    - 地形：通过 SiriusTwoSpanBridgeCfg.terrain (bridge_terrain) 生成
      包含：出发区平台 + 第一段间距 0.05 m 的木板 + 中间平台 +
            第二段间距 0.15 m 的木板 + 终点平台。
    - 机器人动力学、控制、奖励等逻辑：沿用平地 SiriusJoyFlat 的实现。
    - 初始状态：
        * 当前保持和 SiriusFlatCfg 一致。
        * 你可以在本文件中增加 SiriusTwoSpanBridgeCfg.init_state 覆盖，
          或者在 sirius_joystick.SiriusJoyFlat._reset_root_states 里改，
          把起始点限制在“出发平台顶面中心半径 0.25 m 的圆内”。
    """

    def __init__(self, cfg: SiriusTwoSpanBridgeCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
