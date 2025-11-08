# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class SiriusFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        # 默认使用 256 个环境（与 camera.max_envs 匹配）
        # 确保所有环境都有专属相机，避免轮转导致的时序不一致
        # 16GB 显存下，256 envs × 40 MB/camera ≈ 10 GB 相机显存（安全）
        num_envs = 256
        num_actions = 12
        proprio_obs_dim = 45
        depth_obs_width = 87
        depth_obs_height = 58
        num_observations = proprio_obs_dim + depth_obs_width * depth_obs_height

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.445] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "LF_HAA": 0.0,
            "LH_HAA": 0.0,
            "RF_HAA": 0.0,
            "RH_HAA": 0.0,

            "LF_HFE": 0.8,
            "LH_HFE": -0.8,
            "RF_HFE": 0.8,
            "RH_HFE": -0.8,

            "LF_KFE": -1.6,
            "LH_KFE": 1.6,
            "RF_KFE": -1.6,
            "RH_KFE": 1.6,
        }
            
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'HAA': 40., 'HFE': 40., 'KFE': 40.}  # [N*m/rad]
        damping = {'HAA': 2., 'HFE': 2., 'KFE': 2.}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5
        use_actuator_network = False #True
        # actuator_net_file = "{LEGGED_GYM_ROOT_DIR}/resources/actuator_nets/anydrive_v3_lstm.pt"

    class asset( LeggedRobotCfg.asset ):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/sirius_diff_release/urdf/sirius_diff_new.urdf"
        name = "sirius"
        foot_name = "FOOT"
        penalize_contacts_on = ["calf", "thigh"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 #1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False

    class commands( LeggedRobotCfg.commands ):
        heading_command = False
        resampling_time = 4.
        # 更保守的 command curriculum 设置，降低初始目标速度并减小最大扩展
        curriculum = True
        max_curriculum = 0.6  # 降低最大命令幅度，避免过快目标导致摔倒
        # 更保守的扩展步长与门槛（可覆盖 base 默认）
        curriculum_increment = 0.01
        curriculum_progress_threshold = 0.9
        curriculum_consecutive_successes = 10
        curriculum_min_resets_between_expansions = 50
        class ranges( LeggedRobotCfg.commands.ranges ):
            # 初始速度范围更小（鼓励先学会站立和平衡）
            lin_vel_x = [-0.03, 0.03] # min max [m/s]
            lin_vel_y = [-0.03, 0.03]   # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]    # min max [rad/s]，较窄
            heading = [-3.14, 3.14]

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_base_mass = True
        added_mass_range = [-5., 5.]
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
  
    class rewards( LeggedRobotCfg.rewards ):
        base_height_target = 0.445
        max_contact_force = 350
        only_positive_rewards = True
        soft_dof_vel_limit = 0.8
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # 增强对侧倾/姿态错误的惩罚，鼓励直立
            orientation = -20.0
            feet_air_time = 3.
            # 更强的摔倒/高度偏离惩罚（防止跌落）
            base_height = -800.
            posture = 1.0
            # dof_vel = -0.005
            # dof_vel_limits = -0.1
    
    class noise( LeggedRobotCfg.noise ):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.5
            gravity = 0.05
            height_measurements = 0.1

    class sim ( LeggedRobotCfg.sim ):
        dt =  0.004

    # camera defaults for sirius
    class camera(LeggedRobotCfg.camera):
        """深度相机默认配置。设置 enable=True 可开启相机。"""
        enable = True
        body_name = "trunk"  # main body link in sirius URDF
        position = [0.45, 0.0, -0.03]  # forward 45cm, height -3cm
        rpy = [0.0, 0.6, 0.0]  # pitch down ~34 degrees (positive = looking down)
        width = 87
        height = 58
        horizontal_fov = 90.0
        max_depth = 5.0
        
        # ============ 相机刷新频率配置 ============
        # 物理频率: 250 Hz (dt=0.004)
        # 控制频率: 50 Hz (decimation=5, 每 0.02s 一个控制 step)
        # 相机帧率: 30 Hz (真实相机帧率)
        #
        # 计算：50 Hz / 30 Hz ≈ 1.67
        # 取整：每 2 个控制 step 刷新一次 = 25 Hz（接近 30 Hz）
        # 
        # 关键修复：禁用轮转机制，所有环境同步更新！
        obs_refresh_interval = 2  # 每 2 个控制 step (0.04s) 刷新 = 25 Hz
        
        # ============ 显存优化配置 ============
        # 由于需要所有环境同步更新，max_envs 必须 >= num_envs
        # 否则会导致不同环境更新频率不一致（破坏训练稳定性）
        #
        # 16GB 显存限制下的配置策略：
        #   选项 A：少量环境 + 高分辨率（推荐初期调试）
        #     num_envs=256, max_envs=256, width=87×58
        #     显存：3.0 (基础) + 0.6 (envs) + 10.2 (256 cameras) = 13.8 GB
        #
        #   选项 B：中等环境 + 标准分辨率（推荐训练）
        #     num_envs=512, max_envs=512, width=87×58
        #     显存：3.0 + 1.3 + 20.5 = 24.8 GB ❌ 超出 16GB
        #     需要降低分辨率！
        #
        #   选项 C：中等环境 + 降低分辨率（平衡方案）
        #     num_envs=512, max_envs=512, width=58×43
        #     显存：3.0 + 1.3 + 12.8 = 17.1 GB ❌ 仍然超出
        #     进一步降低分辨率或减少环境数
        #
        #   选项 D：少量环境 + 标准分辨率（当前推荐）
        #     num_envs=256, max_envs=256, width=87×58
        #     显存：约 13.8 GB ✅ 安全
        #
        # 重要：如果使用轮转机制（max_envs < num_envs），会导致训练不稳定！
        # 建议：先用 256 envs 训练到稳定，再考虑扩展
        max_envs = 256  # 设置为与 num_envs 相同，禁用轮转
        
        # 启用 GPU tensor 模式（避免 GPU→CPU→GPU 数据传输）
        enable_tensors = True
        
        # 控制是否打印周期性性能统计信息（例如 [Perf] 行）。默认关闭，
        # 仅在你明确希望调试时把 cfg.camera.print_perf 设为 True。
        print_perf = False

class SiriusFlatCfgPPO( LeggedRobotCfgPPO ):
    class policy( LeggedRobotCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        proprio_obs_dim = SiriusFlatCfg.env.proprio_obs_dim
        vision_shape = (1, SiriusFlatCfg.env.depth_obs_height, SiriusFlatCfg.env.depth_obs_width)
        vision_latent_dim = 32
        critic_proprio_obs_dim = SiriusFlatCfg.env.proprio_obs_dim

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # ============ 针对 256 envs 优化的超参数 ============
        # 原始配置是为 4096 envs 设计的，batch_size = 4096×24/4 = 24,576
        # 当前 256 envs: batch_size = 256×32/4 = 2,048（缩小 12 倍）
        
        # 学习率调整：基于 batch_size 缩放规则
        # lr ∝ sqrt(batch_size)
        # 256 envs: lr = 1e-3 × sqrt(2048/98304) = 1e-3 × 0.144 ≈ 1.5e-4
        # 考虑到 epochs 增加，学习率稍微提高到 2e-4
        learning_rate = 2.0e-4  # 从 1e-3 降低到 2.0e-4
        
        # 提高训练稳定性
        clip_param = 0.15       # 从 0.2 降低（更保守的策略更新）
        entropy_coef = 0.02     # 从 0.01 提高（鼓励探索）
        # 适度的训练轮数，平衡稳定性与学习耗时
        num_learning_epochs = 6  # 从 10 降为 6，显著减少学习阶段耗时
        # 减少每个 epoch 的 mini-batch 次数以进一步降低学习时间
        num_mini_batches = 2  # 从默认 4 降为 2（更大 batch，更少反向传播步数）
        # KL 散度约束（adaptive lr 会根据此调整学习率）
        desired_kl = 0.008      # 从 0.01 降低（更保守）
        # 梯度裁剪保持不变（防止梯度爆炸）
        max_grad_norm = 1.0

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = "sirius_diff_release"
        load_run = -1
        
        # 增加总训练步数（因为每次迭代样本少了）
        # 原始 4096 envs × 1200 iters = 4,915,200 步
        # 256 envs × 32 steps × 2400 iters ≈ 19,660,800 总样本
        # 配合 epochs=10，有效样本 ≈ 196M（充分训练）
        max_iterations = 2000  # 从 1200 提高到 2000
        
        # 平衡显存和性能：适度增加步数
        num_steps_per_env = 32  # 从 24 提高到 32（而非 48，节省显存）
        # 新的 batch_size = 256 × 32 / 4 = 2,048
        # 配合 epochs=10，每次迭代实际看到 2048×10 = 20,480 样本
