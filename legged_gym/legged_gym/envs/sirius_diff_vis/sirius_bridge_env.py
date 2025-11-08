# sirius_two_span_bridge.py

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.sirius_diff_vis.sirius_flat_config import SiriusFlatCfg, SiriusFlatCfgPPO
from legged_gym.envs.sirius_diff_vis.sirius_joystick import SiriusJoyFlat


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
        episode_length_s = 30.0

        num_envs = 1

    class commands(SiriusFlatCfg.commands):
        # 开启 heading 模式，让航向速度用目标 heading - 当前 heading 计算
        heading_command = True
        # 保持 resampling 时间一致，并额外提供一个可调的噪声幅度用于轻量域随机化
        resampling_time = SiriusFlatCfg.commands.resampling_time
        lateral_correction_gain = 1.0
        heading_noise_std = 0.05

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
        # display every 3 simulation steps (this is in sim steps, not seconds)
        display_interval_steps = 3
        # window name shown by OpenCV
        display_window_name = "sirius_diff_vis_cam"
        # max depth clipping for normalized depth interface (meters)
        max_depth = 10.0


class SiriusTwoSpanBridgeCfgPPO(SiriusFlatCfgPPO):
    """PPO 配置：沿用 flat 任务的网络尺寸以便顺利从 Stage A checkpoint 恢复。"""

    class policy(SiriusFlatCfgPPO.policy):
        # 继承 flat 任务的网络结构，确保 staged 训练时模型尺寸一致
        pass

    class runner(SiriusFlatCfgPPO.runner):
        experiment_name = "sirius_two_span_bridge"
        max_iterations = 1200


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
