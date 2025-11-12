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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import math

import torch
from torch import Tensor
from typing import Tuple, Dict
try:
    import cv2
except Exception:
    cv2 = None

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .sirius_flat_config import SiriusFlatCfg

class SiriusJoyFlat(BaseTask):
    def __init__(self, cfg: SiriusFlatCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        # Disable debug visualization by default to improve performance when cameras are enabled.
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        # initialize camera-related flags early (create_sim will call _create_envs)
        self._camera_initialized = False
        # handles to cameras in each environment
        self.camera_handles: list = []
        # camera properties used when creating cameras
        self.camera_props = None
        # flag to avoid repeated warnings when target body is missing
        self._camera_warned_missing_body = False
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        # update camera display at a lower rate when running with a viewer
        if not self.headless and getattr(self, '_camera_initialized', False):
            try:
                self._maybe_update_camera_display()
            except Exception:
                pass

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        
        Note: 本体感觉观测存储在 self.obs_buf 中（45维），
              深度图像存储在 self.depth_obs_buf 中（B, 1, H, W）。
              PPO算法会使用这两个buffer来构建完整的观测（45+32=77维）。
              
              ⚠️ 重要：高度测量 (measure_heights) 仅用于奖励计算，不输入模型！
              这样确保训练和部署时模型输入维度一致（部署时无法获取地形高度）。
        """
        # 本体感觉观测 (proprioception): 45维
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel, # 3dim
                                    self.projected_gravity, # 3dim
                                    self.commands[:, :3] * self.commands_scale, # 3dim
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos, # 12dim
                                    self.dof_vel * self.obs_scales.dof_vel, # 12dim
                                    self.actions # 12dim
                                    ),dim=-1)
        
        # 地形高度测量：仅用于奖励计算，不输入模型（保证部署一致性）
        if self.cfg.terrain.measure_heights:
            # 更新内部状态，用于 _reward_base_height() 等奖励函数
            # 注意：不添加到 obs_buf！
            pass
        
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        # 获取深度图像观测 (vision): (num_envs, H, W) → (num_envs, 1, H, W)
        if getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable and self._camera_initialized:
            try:
                # 获取深度图 (num_envs, H, W)，已经线性化并clip到[0, max_depth]
                depth_images = self.get_camera_depth_images(as_torch=True, return_mask=False)
                
                # Debug: Log camera configuration on first call
                if not hasattr(self, '_camera_obs_logged'):
                    enable_tensors = getattr(self.cfg.camera, 'enable_tensors', False)
                    print(f"[Compute Obs] Camera enable_tensors={enable_tensors}")
                    print(f"[Compute Obs] Depth device={depth_images.device}, shape={depth_images.shape}, dtype={depth_images.dtype}")
                    self._camera_obs_logged = True
                
                # 归一化到 [0, 1]
                max_depth = float(getattr(self.cfg.camera, 'max_depth', 5.0))
                depth_normalized = depth_images / max_depth
                
                # 添加通道维度: (num_envs, H, W) → (num_envs, 1, H, W)
                self.depth_obs_buf = depth_normalized.unsqueeze(1)
                
            except Exception as e:
                # 如果相机未初始化或出错，使用零填充
                if not hasattr(self, '_depth_error_warned'):
                    print(f"[Warning] Failed to get depth images: {e}")
                    print("[Warning] Using zero-filled depth buffer as fallback")
                    self._depth_error_warned = True
                
                # 创建零填充的深度buffer
                h = getattr(self.cfg.camera, 'height', 58)
                w = getattr(self.cfg.camera, 'width', 87)
                self.depth_obs_buf = torch.zeros(
                    self.num_envs, 1, h, w,
                    dtype=torch.float32,
                    device=self.device
                )
        else:
            # 相机未启用，使用零填充
            if not hasattr(self, 'depth_obs_buf'):
                h = getattr(self.cfg.camera, 'height', 58) if hasattr(self.cfg, 'camera') else 58
                w = getattr(self.cfg.camera, 'width', 87) if hasattr(self.cfg, 'camera') else 87
                self.depth_obs_buf = torch.zeros(
                    self.num_envs, 1, h, w,
                    dtype=torch.float32,
                    device=self.device
                )

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: 
              - resample commands
              - compute ang vel command based on heading (if enabled)
              - optionally: steer linear velocity commands to follow bridge centerline
              - compute measured terrain heights
              - randomly push robots (if enabled)
        """
        # 1) 正常的命令重采样（决定速度大小等）
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # 2) 原来的 heading/yaw 命令逻辑
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                -1., 1.
            )

        # 3) 新增：让线速度命令 = “沿桥方向前进 + 根据桥面中线偏差纠偏”
        # ---------------------------------------------------
        # 桥面中线：假设与 env_origins 在 y 轴对齐
        base_pos_world = self.root_states[:, :3]      # [num_envs, 3]
        centerline_y = self.env_origins[:, 1]         # [num_envs]
        d_lat = base_pos_world[:, 1] - centerline_y   # 横向偏移，>0 说明在“上方”

        # 可调参数：纠偏增益（建议先从 0.5 或 1.0 开始）
        # 如果你愿意，也可以放到 cfg 里：
        # k_lat = self.cfg.commands.lateral_correction_gain
        k_lat = 1.0

        # 世界系下的“前进 + 纠偏” 2D 方向：x 始终为 1，y 与偏移成反比
        # v_world_xy ∝ [1, -k * d_lat]
        dir_world_xy = torch.stack(
            [
                torch.ones_like(d_lat),   # x 分量：沿桥方向前进
                -k_lat * d_lat            # y 分量：偏右则往左，偏左则往右
            ],
            dim=1
        )  # [num_envs, 2]

        # 防止全 0，单位化
        dir_world_norm = torch.norm(dir_world_xy, dim=1, keepdim=True) + 1e-6
        dir_world_xy_unit = dir_world_xy / dir_world_norm            # [num_envs, 2]

        # 扩展成 3D，方便用 quat_rotate_inverse 转到机体系
        vec_world = torch.zeros_like(base_pos_world)                 # [num_envs, 3]
        vec_world[:, :2] = dir_world_xy_unit

        # 世界系 -> 机体系
        vec_base = quat_rotate_inverse(self.base_quat, vec_world)    # [num_envs, 3]
        dir_base_xy = vec_base[:, :2]
        dir_base_norm = torch.norm(dir_base_xy, dim=1, keepdim=True) + 1e-6
        dir_base_xy_unit = dir_base_xy / dir_base_norm               # [num_envs, 2]

        # 保留原来的速度模长，只改方向
        speed = torch.norm(self.commands[:, :2], dim=1, keepdim=True)  # [num_envs, 1]

        # 可选：只对“本来就要动”的命令做方向修正
        moving_mask = (speed > 1e-3).squeeze(1)                        # [num_envs]
        self.commands[moving_mask, :2] = speed[moving_mask] * dir_base_xy_unit[moving_mask, :]

        # 4) 保持原本地形高度测量与 push_robots 的逻辑
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()


    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (Tensor[int64] on self.device): Environment ids
        """
        if len(env_ids) == 0:
            return

        # 确保是 long 类型，后面索引用
        env_ids = env_ids.to(dtype=torch.long)

        # ------------------- base position -------------------
        # 这里的逻辑：
        # 1. 先按原来方式，把 base 放到对应 env_origin 上
        # 2. 再在 x 方向整体平移，把“中间休息区的中心”移回“出发区中心”
        # 3. 最后在出发区中心半径 0.25 m 的圆内随机一个落点
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # 从“中间休息区”移动到“出发区”的偏移量（沿吊桥方向 x 轴）
        # 几何长度：start(1m) + bridge1(3.05m) + half rest(0.5m) = 4.05m
        # env_origins 当前在中间休息区中心，因此往负 x 平移 4m 回到出发区中心
        bridge_offset_x = -4  # m
        self.root_states[env_ids, 0] += bridge_offset_x

        # 在出发区中心半径 0.25 m 的圆内随机一个 xy 偏移
        # 使用极坐标采样：r = R * sqrt(u), theta = 2πu，保证在圆面上均匀
        R = 0.05
        r = R * torch.sqrt(torch.rand(len(env_ids), 1, device=self.device))
        theta = 2.0 * np.pi * torch.rand(len(env_ids), 1, device=self.device)
        offset_xy = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        self.root_states[env_ids, :2] += offset_xy

        # ------------------- base velocities -------------------
        # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )

        # ------------------- env -> actor indices -------------------
        # robot actor id = env_id * actors_per_env
        actor_ids = (env_ids * int(self.actors_per_env)).to(dtype=torch.long)
        actor_ids_int32 = actor_ids.to(dtype=torch.int32)

        # ------------------- 更新本地 actor buffer -------------------
        self._actor_root_states[actor_ids] = self.root_states[env_ids]

        # ------------------- 同步到仿真 -------------------
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._actor_root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )


    def _push_robots(self):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        max_vel = self.cfg.domain_rand.max_push_vel_xy

        # 给所有 env 的 base x/y 线速度一个随机冲量
        self.root_states[:, 7:9] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device
        )

        # ------------------- env / actor 索引 -------------------
        # 假设每个 env 只有 1 个 robot actor：
        #   robot_actor_id = env_id * actors_per_env
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        actor_ids_long = (env_ids * int(self.actors_per_env)).long()
        actor_ids_int32 = actor_ids_long.to(torch.int32)

        # ------------------- 更新本地 actor buffer -------------------
        # root_states 是 _actor_root_states 的 view，必须 clone 打断别名
        self._actor_root_states[actor_ids_long] = self.root_states.clone()

        # ------------------- 同步到仿真 -------------------
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._actor_root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32),
        )

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    # ---------- camera helpers ----------
    def _quat_from_euler(self, roll: float, pitch: float, yaw: float) -> gymapi.Quat:
        """Create quaternion from ZYX (roll, pitch, yaw) in radians.
        Returns gymapi.Quat(x, y, z, w)."""
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return gymapi.Quat(x, y, z, w)

    def _create_and_attach_cameras(self, body_names):
        """Create a depth camera for each environment and attach it to the specified body.
        This function is called once during environment creation when cameras are enabled in the config.

        Args:
            body_names (List[str]): List of body names from the robot asset.
        """
        # prepare common camera properties
        cam_props = gymapi.CameraProperties()
        cam_props.width = int(self.cfg.camera.width)
        cam_props.height = int(self.cfg.camera.height)
        cam_props.horizontal_fov = float(self.cfg.camera.horizontal_fov)
        cam_props.use_collision_geometry = bool(getattr(self.cfg.camera, 'use_collision_geometry', False))
        # ⚠️ CRITICAL: Must enable tensors here for GPU tensor API to work!
        cam_props.enable_tensors = bool(getattr(self.cfg.camera, 'enable_tensors', False))
        self.camera_props = cam_props
        
        # Debug: Log camera properties
        print(f"[Camera Init] enable_tensors={cam_props.enable_tensors}, use_collision_geometry={cam_props.use_collision_geometry}")

        # find body index to attach to
        target_body_name = getattr(self.cfg.camera, 'body_name', None)
        if target_body_name is not None and target_body_name not in body_names:
            if not self._camera_warned_missing_body:
                print(f"[Camera] body_name '{target_body_name}' not found in asset bodies. Falling back to first body: '{body_names[0]}'")
                self._camera_warned_missing_body = True
            target_body_name = body_names[0]
        if target_body_name is None:
            # prefer trunk on sirius if present
            target_body_name = 'trunk' if 'trunk' in body_names else body_names[0]

        # local transform relative to body
        px, py, pz = self.cfg.camera.position
        r, p, y = self.cfg.camera.rpy
        local_tf = gymapi.Transform()
        local_tf.p = gymapi.Vec3(px, py, pz)
        local_tf.r = self._quat_from_euler(r, p, y)

        self.camera_handles = []
        for i in range(self.num_envs):
            env = self.envs[i]
            cam_h = self.gym.create_camera_sensor(env, cam_props)
            body_handle = self.gym.find_actor_rigid_body_handle(env, self.actor_handles[i], target_body_name)
            if body_handle < 0:
                if not self._camera_warned_missing_body:
                    print(f"[Camera] Could not find body '{target_body_name}'. Attaching to first body index 0.")
                    self._camera_warned_missing_body = True
                body_handle = 0
            self.gym.attach_camera_to_body(cam_h, env, body_handle, local_tf, gymapi.FOLLOW_TRANSFORM)
            self.camera_handles.append(cam_h)
        self._camera_initialized = True

        # Store camera body info for visualization
        self._camera_body_name = target_body_name

    def get_camera_depth_images(self, as_torch: bool = True, return_mask: bool = False):
        """Render and return stacked depth images from all env cameras.
        Returns a tensor/ndarray of shape (num_envs, H, W). Depth is in meters.
        
        GPU-optimized version: Uses GPU tensor API when cfg.camera.enable_tensors=True
        to avoid CPU-GPU transfers.
        """
        if not (getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable and self._camera_initialized):
            raise RuntimeError("Camera is not enabled or not initialized. Set cfg.camera.enable=True before creating the env.")

        # ⚠️ 关键修复：在渲染相机前刷新刚体状态，确保相机位置同步到最新的 trunk 位置
        # 这解决了相机画面滞后于实际机器人位置的问题
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        # Ensure graphics are stepped and sensors rendered
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        H = int(self.camera_props.height)
        W = int(self.camera_props.width)
        max_depth = float(getattr(self.cfg.camera, 'max_depth', getattr(self.cfg.camera, 'far_plane', 10.0)))
        created = len(getattr(self, 'camera_handles', []))
        
        # Check if we should use GPU tensor API (faster, no CPU-GPU transfer)
        use_tensor_api = getattr(self.cfg.camera, 'enable_tensors', False) and as_torch
        
        # Debug: Print which path is being used (only once)
        if not hasattr(self, '_depth_path_logged'):
            enable_tensors_config = getattr(self.cfg.camera, 'enable_tensors', False)
            print(f"[Camera Debug] enable_tensors={enable_tensors_config}, as_torch={as_torch}, use_tensor_api={use_tensor_api}")
            print(f"[Camera Debug] Using {'GPU tensor path (FAST) ✅' if use_tensor_api else 'CPU/NumPy path (SLOW) ⚠️'}")
            self._depth_path_logged = True
        
        if use_tensor_api:
            # ========== GPU Tensor Path (Optimized) ==========
            import torch as _torch
            from isaacgym import gymtorch
            
            # Get camera image tensors directly on GPU (zero-copy!)
            imgs = []
            for i in range(self.num_envs):
                if i < created:
                    # Get tensor directly on GPU (no CPU transfer!)
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH
                    )
                    # Wrap as PyTorch tensor (zero-copy view)
                    depth_torch = gymtorch.wrap_tensor(depth_tensor)
                    imgs.append(depth_torch)
                else:
                    # Placeholder for envs without cameras
                    device = self.device if hasattr(self, 'device') else 'cuda:0'
                    imgs.append(_torch.full((H, W), max_depth, dtype=_torch.float32, device=device))
            
            # Stack on GPU (no CPU involved!)
            arr_raw = _torch.stack(imgs, dim=0)  # [num_envs, H, W]
            
            # Process on GPU
            arr_linear = arr_raw.clone()
            hit_mask = _torch.isfinite(arr_raw)
            
            # Check if normalized depth [0,1] (use torch operations that work in PyTorch 1.13)
            finite_mask = _torch.isfinite(arr_raw)
            if finite_mask.any():
                arr_max = arr_raw[finite_mask].max().item()
                arr_min = arr_raw[finite_mask].min().item()
            else:
                arr_max = 1.0
                arr_min = 0.0
            
            if arr_max <= 1.01 and arr_min >= -0.01:
                # Convert normalized depth to linear depth
                near = float(getattr(self.cfg.camera, 'near_plane', getattr(self.cfg.camera, 'near', 0.05)))
                far = float(getattr(self.cfg.camera, 'far_plane', getattr(self.cfg.camera, 'far', 10.0)))
                
                ndc = arr_raw * 2.0 - 1.0
                denom = (far + near - ndc * (far - near))
                
                # Avoid division by zero
                mask = _torch.abs(denom) > 1e-6
                z = _torch.zeros_like(arr_raw)
                z[mask] = (2.0 * near * far) / denom[mask]
                arr_linear = _torch.abs(z)
            else:
                arr_linear = _torch.abs(arr_raw)
            
            # Handle infinities and clip
            arr_linear[~_torch.isfinite(arr_linear)] = max_depth
            arr_linear = _torch.clamp(arr_linear, 0.0, max_depth)
            
            if return_mask:
                return arr_linear, hit_mask
            return arr_linear
            
        else:
            # ========== CPU/NumPy Path (Legacy, for compatibility) ==========
            import numpy as _np
            
            imgs = []
            for i in range(self.num_envs):
                if i < created:
                    depth = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_DEPTH)
                    imgs.append(depth.astype('float32'))
                else:
                    imgs.append(_np.full((H, W), max_depth, dtype='float32'))
            
            # Stack on CPU
            arr_raw = _np.stack(imgs, axis=0)
            
            # Process on CPU
            arr_linear = arr_raw.copy().astype('float32')
            hit_mask = _np.isfinite(arr_raw)
            
            try:
                arr_max = float(_np.nanmax(arr_raw))
                arr_min = float(_np.nanmin(arr_raw))
            except Exception:
                arr_max = 1.0
                arr_min = 0.0
            
            if arr_max <= 1.01 and arr_min >= -0.01:
                near = float(getattr(self.cfg.camera, 'near_plane', getattr(self.cfg.camera, 'near', 0.05)))
                far = float(getattr(self.cfg.camera, 'far_plane', getattr(self.cfg.camera, 'far', 10.0)))
                ndc = arr_raw * 2.0 - 1.0
                denom = (far + near - ndc * (far - near))
                with _np.errstate(divide='ignore', invalid='ignore'):
                    z = (2.0 * near * far) / denom
                arr_linear = _np.abs(z.astype('float32'))
            else:
                with _np.errstate(invalid='ignore'):
                    arr_linear = _np.abs(arr_raw)
            
            arr_linear[~_np.isfinite(arr_linear)] = max_depth
            arr_linear = _np.clip(arr_linear, 0.0, max_depth)
            
            if as_torch:
                import torch as _torch
                t = _torch.from_numpy(arr_linear)
                t = t.to(_torch.get_default_dtype())
                device = self.device if hasattr(self, 'device') else 'cpu'
                t = t.to(device)
                
                if return_mask:
                    m = _torch.from_numpy(hit_mask.astype('bool')).to(device)
                    return t, m
                return t
            else:
                if return_mask:
                    return arr_linear, hit_mask
                return arr_linear

    def get_camera_rgb_images(self, as_torch: bool = False, to_bgr: bool = True):
        """Render and return stacked RGB images from all env cameras.
        Returns array/tensor of shape (num_envs, H, W, 3) in RGB order by default. If to_bgr=True and returning numpy, converts to BGR for OpenCV display.
        """
        if not (getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable and self._camera_initialized):
            raise RuntimeError("Camera is not enabled or not initialized. Set cfg.camera.enable=True before creating the env.")

        # ⚠️ 同样需要刷新刚体状态（用于 OpenCV 可视化显示）
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        imgs = []
        for i in range(self.num_envs):
            img = self.gym.get_camera_image(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            img = img[:, :, :3].astype('uint8')
            imgs.append(img)
        import numpy as _np
        arr = _np.stack(imgs, axis=0)
        if as_torch:
            import torch as _torch
            t = _torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
            try:
                device = self.device if hasattr(self, 'device') else 'cpu'
                t = t.to(device)
            except Exception:
                pass
            return t
        if to_bgr and cv2 is not None:
            arr = arr[:, :, :, ::-1]
        return arr

    def _maybe_update_camera_display(self):
        """Optionally opens an OpenCV window and displays the first env camera at a lower rate."""
        if self.headless or not getattr(self, '_camera_initialized', False) or cv2 is None:
            return
        if not (hasattr(self, 'viewer') and self.viewer is not None and self.enable_viewer_sync):
            return
        # set up counters on first use
        if not hasattr(self, '_camera_display_counter'):
            self._camera_display_counter = 0
            self.camera_display_interval_steps = int(getattr(self.cfg.camera, 'display_interval_steps', 3))
            self._camera_display_window_name = getattr(self.cfg.camera, 'display_window_name', 'env_cameras')

        self._camera_display_counter = (self._camera_display_counter + 1) % max(1, self.camera_display_interval_steps)
        if self._camera_display_counter != 0:
            return

        try:
            imgs = self.get_camera_rgb_images(as_torch=False, to_bgr=True)
        except Exception:
            return
        img = imgs[0]
        max_width = 960
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            arr_raw = _np.stack(imgs, axis=0)

            # prepare linearized depth and hit mask
            arr_linear = arr_raw.copy().astype('float32')
            hit_mask = _np.isfinite(arr_raw)

            try:
                arr_max = float(_np.nanmax(arr_raw))
                arr_min = float(_np.nanmin(arr_raw))
            except Exception:
                arr_max = 1.0
                arr_min = 0.0

        cv2.imshow(self._camera_display_window_name, img)
        cv2.waitKey(1)

    def visualize_camera_position(self):
        """Draw camera position and viewing direction in the viewer.
        Call this after step() is called to see where the camera is mounted.
        This uses the tensor API which is compatible with GPU pipeline.
        """
        if not (getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable and self._camera_initialized):
            print("[Camera] Camera not initialized, cannot visualize")
            return


            arr_linear[~_np.isfinite(arr_linear)] = max_depth
            arr_linear = _np.clip(arr_linear, 0.0, max_depth)

            # save debug artifacts if possible
            try:
                import os as _os
                out_dir = _os.path.join('/home', 'eziothean', 'Sirius_RL_Gym-master', 'legged_gym', 'legged_gym', 'scripts', 'camera_outputs')
                _np.save(_os.path.join(out_dir, 'depth_raw_renderer_latest.npy'), arr_raw[0] if arr_raw.shape[0] == 1 else arr_raw)
                _np.save(_os.path.join(out_dir, 'depth_linearized_latest.npy'), arr_linear[0] if arr_linear.shape[0] == 1 else arr_linear)
                _np.save(_os.path.join(out_dir, 'depth_mask_latest.npy'), hit_mask[0] if hit_mask.shape[0] == 1 else hit_mask)
            except Exception:
                pass

            arr = arr_linear
        from isaacgym import gymutil
        # Get camera configuration
        px, py, pz = self.cfg.camera.position
        r, p, yaw = self.cfg.camera.rpy

        # Draw for first few environments only (to avoid clutter)
        num_envs_to_draw = min(4, self.num_envs)

        # Use tensor API to get root states
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_positions = self.root_states[:, :3].cpu().numpy()  # (N, 3)

        for i in range(num_envs_to_draw):
            env = self.envs[i]
            # Get robot base position from tensor
            base_x, base_y, base_z = root_positions[i]

            # Camera position in world frame (simplified - relative to robot base)
            cam_world_x = float(base_x) + px
            cam_world_y = float(base_y) + py
            cam_world_z = float(base_z) + pz

            # Draw camera origin as a sphere (RED)
            sphere_geom = gymutil.WireframeSphereGeometry(0.05, 12, 12, None, color=(1, 0, 0))
            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(cam_world_x, cam_world_y, cam_world_z)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env, sphere_pose)

            # Draw camera viewing direction (RED arrow - forward)
            forward_length = 0.4
            forward_x = forward_length * math.cos(p)
            forward_z = -forward_length * math.sin(p)

            line_verts = [
                [cam_world_x, cam_world_y, cam_world_z],
                [cam_world_x + forward_x, cam_world_y, cam_world_z + forward_z]
            ]
            line_colors = [[1, 0, 0], [1, 0.5, 0]]  # Red to orange
            self.gym.add_lines(self.viewer, env, 1, line_verts, line_colors)

            # Draw side indicators (GREEN for left/right)
            side_length = 0.2
            line_verts = [
                [cam_world_x, cam_world_y - side_length, cam_world_z],
                [cam_world_x, cam_world_y + side_length, cam_world_z]
            ]
            line_colors = [[0, 1, 0], [0, 1, 0]]
            self.gym.add_lines(self.viewer, env, 1, line_verts, line_colors)

            # Draw up indicator (BLUE)
            line_verts = [
                [cam_world_x, cam_world_y, cam_world_z],
                [cam_world_x, cam_world_y, cam_world_z + 0.2]
            ]
            line_colors = [[0, 0.5, 1], [0, 0.5, 1]]
            self.gym.add_lines(self.viewer, env, 1, line_verts, line_colors)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:45] = 0. # previous actions
        # 注意：高度测量不再输入模型，因此不需要噪声
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        # Note: the actor root state buffer can contain more actors than robots
        # (e.g., terrain, other assets). Wrap the full buffer and then take
        # the first `self.num_envs` rows as the robot root states to ensure
        # consistency between buffers that are per-robot.
        root_states_all = gymtorch.wrap_tensor(actor_root_state)
        num_total_actors = root_states_all.shape[0]
        assert num_total_actors >= self.num_envs
        self.actors_per_env = num_total_actors // self.num_envs
        self._actor_root_states = root_states_all
        self.root_states = root_states_all[:self.num_envs]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        # contact forces may also be reported per actor, so wrap and slice similarly
        contact_forces_all = gymtorch.wrap_tensor(net_contact_forces).view(num_total_actors, -1, 3)
        self.contact_forces = contact_forces_all[:self.num_envs] # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # ensure gravity and forward vectors have the same leading batch dim as base_quat
        g = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device)
        f = to_torch([1., 0., 0.], device=self.device)
        # use base_quat shape (actual wrapped root_states batch size) to avoid accidental mismatches
        batch = self.base_quat.shape[0]
        self.gravity_vec = g.unsqueeze(0).expand(batch, -1).contiguous()
        self.forward_vec = f.unsqueeze(0).expand(batch, -1).contiguous()
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
        # debug: print resolved asset path when camera is enabled to diagnose body name mismatches
        if getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable:
            print(f"[Camera Debug] Resolved asset path (sirius_joystick): {asset_path}")

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # debug: print a short list of body names when camera is enabled
        if getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable:
            try:
                print(f"[Camera Debug] Loaded asset body names (first 50): {body_names[:50]}")
            except Exception:
                print(f"[Camera Debug] Loaded asset has {len(body_names)} bodies")
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        # optionally create and attach depth cameras
        if getattr(self.cfg, 'camera', None) is not None and self.cfg.camera.enable:
            self._create_and_attach_cameras(body_names)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            
            # 🐛 调试输出：打印课程学习配置
            print(f"\n{'='*60}")
            print(f"[Terrain Initialization] 地形课程学习配置")
            print(f"{'='*60}")
            print(f"  curriculum 启用: {self.cfg.terrain.curriculum}")
            print(f"  max_init_terrain_level (配置): {self.cfg.terrain.max_init_terrain_level}")
            print(f"  max_init_level (实际使用): {max_init_level}")
            print(f"  num_rows (总难度数): {self.cfg.terrain.num_rows}")
            print(f"  num_cols (地形类型): {self.cfg.terrain.num_cols}")
            print(f"  num_envs (环境数): {self.num_envs}")
            
            # 🎨 混合策略：大部分环境课程学习 + 少量环境视觉探索
            visual_exploration_ratio = getattr(self.cfg.terrain, 'visual_exploration_ratio', 0.0)
            num_curriculum_envs = int(self.num_envs * (1 - visual_exploration_ratio))
            num_exploration_envs = self.num_envs - num_curriculum_envs
            
            # 课程学习环境：在 [0, max_init_level] 范围内随机
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            
            # 视觉探索环境：在所有难度随机分布（提供视觉多样性）
            if num_exploration_envs > 0:
                exploration_ids = torch.randperm(self.num_envs, device=self.device)[:num_exploration_envs]
                self.terrain_levels[exploration_ids] = torch.randint(
                    0, self.cfg.terrain.num_rows, 
                    (num_exploration_envs,), 
                    device=self.device
                )
                print(f"  visual_exploration_ratio: {visual_exploration_ratio:.1%}")
                print(f"  课程学习环境: {num_curriculum_envs} ({num_curriculum_envs/self.num_envs:.1%})")
                print(f"  视觉探索环境: {num_exploration_envs} ({num_exploration_envs/self.num_envs:.1%})")
            
            # 🐛 打印初始地形难度分布
            print(f"\n初始地形难度分布:")
            for level in range(self.cfg.terrain.num_rows):
                count = (self.terrain_levels == level).sum().item()
                percentage = count / self.num_envs * 100
                marker = "📚" if level <= max_init_level else "🎨"  # 课程学习 vs 视觉探索
                print(f"  {marker} 难度 {level}: {count:4d} 个环境 ({percentage:5.2f}%)")
            print(f"{'='*60}\n")
            
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """Draw simple debug viz:
           - 蓝色方块：命令向量尾部（起点，在狗头上方）
           - 红色方块：命令向量头部（终点，表示指向方向）
           - 可选：地形高度采样点
        """
        if self.viewer is None:
            return

        # 清空之前画的线框
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # ============== 1) 计算命令向量在世界坐标系下的方向 ==============
        # commands[:, :2] 在机体系，扩成 3D 再用 yaw 旋到世界系
        cmd_body = torch.zeros(self.num_envs, 3, device=self.device)
        cmd_body[:, :2] = self.commands[:, :2]

        cmd_world = quat_apply_yaw(self.base_quat, cmd_body).cpu().numpy()   # [N,3]
        base_pos_all = self.root_states[:, :3].cpu().numpy()                 # [N,3]

        arrow_len = 1.0   # 命令方向可视化长度（m）
        z_offset = 0.25    # 提到狗头上方一点
        max_envs_to_draw = min(self.num_envs, 16)

        # 两个几何体：蓝色尾，红色头
        tail_box = gymutil.WireframeBoxGeometry(
            0.15, 0.15, 0.15, None, color=(0, 0, 1)   # 蓝色
        )
        head_box = gymutil.WireframeBoxGeometry(
            0.15, 0.15, 0.15, None, color=(1, 0, 0)   # 红色
        )

        for i in range(max_envs_to_draw):
            # 起点：机体位置 + z 偏移
            p_base = base_pos_all[i].copy()
            p_base[2] += z_offset

            v = cmd_world[i].copy()
            v[2] = 0.0       # 只关心水平面方向

            norm_xy = np.linalg.norm(v[:2])
            if norm_xy < 1e-3:
                continue     # 命令几乎为 0 就不画

            # 单位化后乘固定长度
            v[:2] = v[:2] / norm_xy * arrow_len

            # 尾部（蓝）：命令起点
            p_tail = p_base
            # 头部（红）：命令指向方向的终点
            p_head = p_base + v * 0.75

            tail_pose = gymapi.Transform(
                gymapi.Vec3(float(p_tail[0]), float(p_tail[1]), float(p_tail[2])),
                gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            )
            head_pose = gymapi.Transform(
                gymapi.Vec3(float(p_head[0]), float(p_head[1]), float(p_head[2])),
                gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            )

            # 画两个方块：蓝尾 + 红头
            gymutil.draw_lines(tail_box, self.gym, self.viewer, self.envs[i], tail_pose)
            gymutil.draw_lines(head_box, self.gym, self.viewer, self.envs[i], head_pose)

        # ============== 2) 可选：画高度采样点（你原来的逻辑） ==============
        if self.terrain.cfg.measure_heights:
            sphere_geom = gymutil.WireframeSphereGeometry(
                0.02, 4, 4, None, color=(1, 1, 0)
            )
            for i in range(self.num_envs):
                base_pos = self.root_states[i, :3].cpu().numpy()
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(
                    self.base_quat[i].repeat(heights.shape[0]),
                    self.height_points[i]
                ).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(
                        sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose
                    )


    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        # 计算机器人相对于当前地形的高度（脚下地形的平均高度）
        terrain_height = torch.mean(self.measured_heights, dim=1)  # [num_envs]
        base_height_above_terrain = self.root_states[:, 2] - terrain_height  # [num_envs]
        
        # 目标是保持在地形上方 base_height_target 的高度
        # 这样在平地、dimps、斜坡等各种地形都能自适应
        return torch.square(base_height_above_terrain - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_posture(self):
        weight = torch.tensor([1.0, 1.0, 0.1] * 4, device=self.device).unsqueeze(0) # shape: (1, num_dof)
        return torch.exp(-torch.sum(torch.square(self.dof_pos - self.default_dof_pos) * weight, dim=1))