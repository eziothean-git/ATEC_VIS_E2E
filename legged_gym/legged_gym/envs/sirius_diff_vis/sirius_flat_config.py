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
from .sirius_shared_model import SiriusSharedPPOCfg

class SiriusFlatCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_actions = 12
        num_observations = 45  # 本体观测维度 (不包含视觉，视觉通过 depth_obs_buf 单独传递)

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
        min_forward_speed = 0.15  # 🔧 最小前进速度（m/s）- 避免采样到过小的速度导致机器人几乎不动
        class ranges( LeggedRobotCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

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
            orientation = -5.0
            feet_air_time = 3.
            base_height = -200.
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
        """深度相机配置 - 用于端到端视觉强化学习"""
        enable = True  # 启用相机（阶段1和阶段2都需要）
        body_name = "trunk"  # main body link in sirius URDF
        position = [0.45, 0.0, -0.03]  # forward 45cm, height -3cm
        rpy = [0.0, 0.6, 0.0]  # pitch down ~34 degrees (positive = looking down)
        
        # 降低分辨率以减少计算负担
        width = 87  # 原 160 -> 87 (约 0.54x)
        height = 58  # 原 120 -> 58 (约 0.48x)
        horizontal_fov = 90.0
        
        # 深度范围
        max_depth = 5.0  # 限制在5米以内（桥梁场景足够）
        near_plane = 0.1
        far_plane = 10.0
        
        # 调试选项（训练时关闭以提高性能）
        debug_outputs = False
        use_collision_geometry = False
        
        # ⚠️ 性能关键：使用 GPU tensor 路径避免 CPU-GPU 传输
        # enable_tensors = True 会直接在 GPU 上生成深度图 tensor
        # 避免 CPU → NumPy → PyTorch → GPU 的转换链
        enable_tensors = True  # ✅ 启用 GPU tensor 路径

class SiriusFlatCfgPPO( LeggedRobotCfgPPO ):
    """
    Sirius 平地任务的 PPO 训练配置。
    
    继承共享模型配置 (SiriusSharedPPOCfg)，确保与 sirius_diff_vis 任务的网络结构一致。
    只覆盖 runner 中的任务特定参数（实验名、迭代次数等）。
    """
    
    # 继承共享的 vision_encoder 配置（视觉编码器）
    class vision_encoder(SiriusSharedPPOCfg.vision_encoder):
        pass
    
    # 继承共享的 policy 配置（网络结构）
    class policy(SiriusSharedPPOCfg.policy):
        # 使用共享配置的 [256, 128, 64]
        # 不要在这里覆盖 actor_hidden_dims 或 critic_hidden_dims！
        pass
    
    # 继承共享的 algorithm 配置（PPO 超参数）
    class algorithm(SiriusSharedPPOCfg.algorithm):
        # 降低学习率到 1/4，因为只能跑 1024 envs 而非 4096
        # 原始: 1.e-3, 现在: 2.5e-4
        learning_rate = 2.5e-4
        
        # GPU 利用率优化：增加学习轮数补偿环境数减少
        # 原始 5 epochs * 4096 envs = 20480 samples/update
        # 现在 8 epochs * 1024 envs = 8192 samples/update (仍少于原始)
        # 但能提高学习阶段的 GPU 利用率
        num_learning_epochs = 8  # 从 5 增加到 8
        
        # 可选：减少 mini_batches 以增加每批的大小
        # mini_batch_size = (num_envs * num_steps_per_env) / num_mini_batches
        # 原始: (4096 * 24) / 4 = 24576
        # 现在: (1024 * 24) / 2 = 12288 (更大的批量利用 GPU)
        num_mini_batches = 2  # 从 4 减少到 2

    # 任务特定的 runner 配置
    class runner(SiriusSharedPPOCfg.runner):
        run_name = ''
        experiment_name = "sirius_flat"  # 阶段1: 平地训练
        load_run = -1
        max_iterations = 1800
