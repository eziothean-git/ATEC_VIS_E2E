# sirius_teacher_curriculum_config.py
"""
Sirius 教师策略 Curriculum 后训练配置

🎯 目标：让教师策略（纯本体感知MLP）在curriculum地形上进行后训练

特点：
- 策略：纯本体感知 MLP（45维输入）
- 相机：关闭（不使用视觉）
- 地形：Curriculum（多种地形类型和难度）
- 用途：
  1. 测试教师策略在复杂地形上的泛化能力
  2. 为IL训练提供更强的教师（如果需要）
  3. 对比有视觉和无视觉策略的性能差异

使用方法：
    # 从平地训练的checkpoint继续
    python train.py --task=sirius_teacher_curriculum --num_envs=4096 --headless \\
        --resume --load_run=logs/sirius_flat/YYYY-MM-DD/HH-MM-SS
    
    # 或从头开始训练
    python train.py --task=sirius_teacher_curriculum --num_envs=4096 --headless
"""

from .sirius_curriculum_config import SiriusCurriculumCfg
from .sirius_joystick import SiriusJoyFlat  # 直接继承基类
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class SiriusTeacherCurriculumCfg(SiriusCurriculumCfg):
    """
    教师策略 Curriculum 环境配置
    
    继承自 SiriusCurriculumCfg，但关闭相机
    """
    
    class env(SiriusCurriculumCfg.env):
        num_envs = 4096  # 教师策略可以用更多env（无视觉渲染开销）
        episode_length_s = 24
        num_observations = 45  # 纯本体观测
    
    class terrain(SiriusCurriculumCfg.terrain):
        """完整的地形课程配置"""
        mesh_type = "trimesh"
        curriculum = True
        selected = False
        measure_heights = True  # 需要测量高度用于奖励计算
        
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 8
        
        # 从较低难度开始（如果从平地checkpoint继续）
        max_init_terrain_level = 1  # 从简单地形开始
        
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
        """相机配置 - 教师策略不使用视觉"""
        enable = False  # ❌ 关闭相机（纯本体感知）
    
    class commands(SiriusCurriculumCfg.commands):
        """命令课程 - 与curriculum保持一致"""
        curriculum = True
        max_curriculum = 0.8
        max_reverse_curriculum = 0.1
        min_forward_speed = 0.2
        curriculum_step = 0.1
        curriculum_threshold = 0.8  # 🔧 降低阈值：tracking reward 达到 80% 即可晋级（原来是0.5，但实际要求太低）
        
        heading_command = True  # 🔥 使用朝向目标而不是角速度，朝向与速度方向对齐
        
        class ranges:
            lin_vel_x = [-0.1, 0.3]
            lin_vel_y = [-0.3, 0.3]
            ang_vel_yaw = [-0.6, 0.6]
            heading = [-3.14, 3.14]
    
    class rewards(SiriusCurriculumCfg.rewards):
        """奖励配置 - 与curriculum保持一致"""
        only_positive_rewards = False
        
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


class SiriusTeacherCurriculumCfgPPO(LeggedRobotCfgPPO):
    """
    教师策略 Curriculum 训练的 PPO 配置
    
    特点：
    - 使用纯本体感知的 ActorCritic（MLP）
    - 不使用视觉编码器
    - 网络结构与平地训练一致
    """
    
    # ========== 不使用视觉配置 ==========
    # 不定义 vision_encoder
    
    # ========== Policy 配置 ==========
    class policy(LeggedRobotCfgPPO.policy):
        """纯本体感知的 MLP 策略"""
        init_noise_std = 1.0
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'
        # 不设置 use_vision
    
    # ========== Algorithm 配置 ==========
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """PPO 超参数"""
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        
        # Curriculum 训练需要更多探索
        entropy_coef = 0.015
        
        # 学习配置
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.e-3  # 标准学习率（4096 envs）
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
    
    # ========== Runner 配置 ==========
    class runner(LeggedRobotCfgPPO.runner):
        """训练运行配置"""
        experiment_name = "sirius_teacher_curriculum"
        run_name = ''
        
        # ⚠️ 使用标准的 ActorCritic（纯 MLP）
        policy_class_name = 'ActorCritic'
        
        # 训练配置
        max_iterations = 2000  # Curriculum 训练需要更多迭代
        num_steps_per_env = 24
        save_interval = 50
        
        # 从平地checkpoint继续（可选）
        # 🔧 支持跨任务加载：可以从 sirius_flat 的checkpoint开始
        resume = False
        load_run = -1  # 设为具体路径如 "sirius_flat/Nov14_20-47-08_" 来跨任务加载
        checkpoint = -1


class SiriusTeacherCurriculum(SiriusJoyFlat):
    """
    教师策略 Curriculum 环境
    
    ⚠️ 继承自 SiriusJoyFlat（基类），而不是 SiriusCurriculum
    原因：避免引入相机相关的逻辑（数据增强课程等）
    
    手动实现：
    - 地形课程更新
    - 命令课程更新  
    - 其他curriculum功能
    """
    
    def __init__(self, cfg: SiriusTeacherCurriculumCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        print("\n" + "="*80)
        print("🔥 [SiriusTeacherCurriculum] Environment initialized")
        print(f"  - Strategy: Pure proprioception (no vision)")
        print(f"  - Terrain: {self.cfg.terrain.mesh_type} (curriculum={self.cfg.terrain.curriculum})")
        print(f"  - Camera: {self.cfg.camera.enable}")
        print(f"  - Num envs: {self.cfg.env.num_envs}")
        print(f"  - Obs dim: {self.cfg.env.num_observations}")
        print(f"  - heading_command: {self.cfg.commands.heading_command}")
        print("="*80 + "\n")
    
    def _post_physics_step_callback(self):
        """
        物理步进后的回调 - 教师课程学习版本
        
        🔧 关键改动：
        1. 移除桥面中线纠偏逻辑（会覆盖速度命令）
        2. 保留基础的命令重采样和朝向转换
        3. 保留地形高度测量和推机器人逻辑
        """
        # 1) 命令重采样
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        # 2) heading_command 模式：将朝向目标转换为角速度命令
        if self.cfg.commands.heading_command:
            import torch
            from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
            
            forward = quat_apply_yaw(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                -1., 1.
            )
            
            # 🐛 调试：每1000步检查朝向转换
            if self.common_step_counter % 1000 == 0:
                sample_size = min(5, self.num_envs)
                sample_ids = torch.arange(sample_size, device=self.device)
                
                print(f"\n{'='*70}")
                print(f"[Heading to AngVel Conversion] Step {self.common_step_counter}")
                print(f"{'='*70}")
                
                for i, env_id in enumerate(sample_ids):
                    target_heading = self.commands[env_id, 3].item()
                    curr_heading = heading[env_id].item()
                    ang_vel_cmd = self.commands[env_id, 2].item()
                    actual_ang_vel = self.base_ang_vel[env_id, 2].item()
                    
                    heading_error = wrap_to_pi(self.commands[env_id, 3] - heading[env_id]).item()
                    
                    print(f"  Env {env_id.item():4d}: "
                          f"target={target_heading*180/3.14159:+6.1f}°, "
                          f"current={curr_heading*180/3.14159:+6.1f}°, "
                          f"error={heading_error*180/3.14159:+6.1f}°, "
                          f"cmd_ω={ang_vel_cmd:+.3f}, "
                          f"actual_ω={actual_ang_vel:+.3f}")
                print(f"{'='*70}\n")

        # 3) 保持地形高度测量
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
        # 4) 保持推机器人逻辑
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    
    def _resample_commands(self, env_ids):
        """
        重采样命令 - 教师课程学习优化
        
        🎯 关键改进：朝向与速度方向对齐（正负5度范围内）
        
        策略：
        1. 速度方向：随机采样 x 和 y 方向
        2. 朝向命令：根据速度方向设置，只在目标方向正负5度范围内
        3. 运动模式：60% 直行，30% 混合，10% 转向为主
        
        Args:
            env_ids (List[int]): 需要重新采样命令的环境ID
        """
        import torch
        from isaacgym.torch_utils import torch_rand_float
        
        if len(env_ids) == 0:
            return
        
        # ============ 1. 线速度 X 方向采样 ============
        direction_selector = torch.rand(len(env_ids), device=self.device)
        forward_mask = direction_selector < 0.85  # 85% 前进
        backward_mask = ~forward_mask  # 15% 后退
        
        # 前进命令
        if forward_mask.any():
            min_forward_speed = getattr(self.cfg.commands, 'min_forward_speed', 0.2)
            max_forward_speed = self.command_ranges["lin_vel_x"][1]
            self.commands[env_ids[forward_mask], 0] = torch_rand_float(
                min_forward_speed, 
                max_forward_speed, 
                (forward_mask.sum(), 1), 
                device=self.device
            ).squeeze(1)
        
        # 后退命令
        if backward_mask.any():
            self.commands[env_ids[backward_mask], 0] = torch_rand_float(
                self.command_ranges["lin_vel_x"][0], 
                0., 
                (backward_mask.sum(), 1), 
                device=self.device
            ).squeeze(1)
        
        # ============ 2. 线速度 Y 方向采样 ============
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0], 
            self.command_ranges["lin_vel_y"][1], 
            (len(env_ids), 1), 
            device=self.device
        ).squeeze(1)
        
        # ============ 3. 朝向命令：与速度方向对齐（±5度）============
        if self.cfg.commands.heading_command:
            # 计算速度方向（atan2(vy, vx)）
            vel_x = self.commands[env_ids, 0]
            vel_y = self.commands[env_ids, 1]
            
            # 计算速度方向角度
            vel_direction = torch.atan2(vel_y, vel_x)
            
            # 在速度方向正负5度（0.0873 rad）范围内随机采样
            heading_offset_range = 5.0 * (3.14159265359 / 180.0)  # 5度转弧度
            heading_offset = torch_rand_float(
                -heading_offset_range,
                heading_offset_range,
                (len(env_ids), 1),
                device=self.device
            ).squeeze(1)
            
            # 设置朝向命令：速度方向 + 小偏移
            self.commands[env_ids, 3] = vel_direction + heading_offset
            
            # 归一化到 [-π, π]
            self.commands[env_ids, 3] = torch.atan2(
                torch.sin(self.commands[env_ids, 3]),
                torch.cos(self.commands[env_ids, 3])
            )
            
            # 处理速度接近零的情况：设置朝向为0（正前方）
            vel_norm = torch.norm(self.commands[env_ids, :2], dim=1)
            zero_vel_mask = vel_norm < 0.1
            if zero_vel_mask.any():
                # 速度很小时，朝向随机在 [-5度, 5度] 范围
                self.commands[env_ids[zero_vel_mask], 3] = torch_rand_float(
                    -heading_offset_range,
                    heading_offset_range,
                    (zero_vel_mask.sum(), 1),
                    device=self.device
                ).squeeze(1)
            
            # 🐛 调试：每1000步检查朝向对齐
            if self.common_step_counter % 1000 == 0 and len(env_ids) > 0:
                sample_size = min(5, len(env_ids))
                sample_env_ids = env_ids[:sample_size]
                
                print(f"\n{'='*70}")
                print(f"[Heading Alignment Check] Step {self.common_step_counter}")
                print(f"{'='*70}")
                for env_id in sample_env_ids:
                    vx = self.commands[env_id, 0].item()
                    vy = self.commands[env_id, 1].item()
                    heading = self.commands[env_id, 3].item()
                    vel_dir = torch.atan2(torch.tensor(vy), torch.tensor(vx)).item()
                    diff = abs(heading - vel_dir)
                    diff = min(diff, 2*3.14159 - diff)  # 处理周期性
                    
                    print(f"  Env {env_id.item():4d}: vel=({vx:+.3f}, {vy:+.3f}), "
                          f"vel_dir={vel_dir*180/3.14159:+6.1f}°, "
                          f"heading={heading*180/3.14159:+6.1f}°, "
                          f"diff={diff*180/3.14159:4.1f}°")
                print(f"{'='*70}\n")
        else:
            # 如果不使用 heading_command，使用角速度
            self.commands[env_ids, 2] = torch_rand_float(
                self.command_ranges["ang_vel_yaw"][0], 
                self.command_ranges["ang_vel_yaw"][1], 
                (len(env_ids), 1), 
                device=self.device
            ).squeeze(1)

        # ============ 4. 运动模式：避免同时高速+高旋转 ============
        mode_selector = torch.rand(len(env_ids), device=self.device)
        
        # 60% 直行为主模式：角速度降低到30%
        straight_mode = mode_selector < 0.6
        if not self.cfg.commands.heading_command:
            self.commands[env_ids[straight_mode], 2] *= 0.3
        
        # 30% 混合模式：都不降低（保持原样）
        # 不修改 mixed_mode 的命令
        
        # 10% 转向为主模式：线速度降低到40%
        turning_mode = mode_selector >= 0.9
        self.commands[env_ids[turning_mode], 0] *= 0.4
        self.commands[env_ids[turning_mode], 1] *= 0.4

        # ============ 5. 过滤过小的命令 ============
        # 将幅值小于 0.15 m/s 的线速度命令置零（避免过小的扰动）
        lin_vel_norm = torch.norm(self.commands[env_ids, :2], dim=1)
        small_cmd_mask = lin_vel_norm < 0.15
        self.commands[env_ids[small_cmd_mask], :2] = 0.0
        
        if self.cfg.commands.heading_command:
            self.commands[env_ids[small_cmd_mask], 3] = 0.0
    
    def _reset_root_states(self, env_ids):
        """
        重置机器人根状态（位置和速度）
        
        🎯 标准的课程学习生成逻辑（与 SiriusCurriculum 相同）
        
        Args:
            env_ids (Tensor[int64]): 需要重置的环境ID
        """
        if len(env_ids) == 0:
            return
        
        import torch
        from isaacgym import gymtorch
        
        # 确保是 long 类型
        env_ids = env_ids.to(dtype=torch.long)
        
        # ------------------- 基础位置设置 -------------------
        # 1. 从初始状态开始
        self.root_states[env_ids] = self.base_init_state
        
        # 2. 添加环境原点偏移（地形中心）
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # 3. 在地形中心附近随机一个位置（半径 0.5m 的圆内）
        R = 0.5  # 半径 (m)
        r = R * torch.sqrt(torch.rand(len(env_ids), 1, device=self.device))
        theta = 2.0 * 3.14159265359 * torch.rand(len(env_ids), 1, device=self.device)
        offset_xy = torch.cat([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
        self.root_states[env_ids, :2] += offset_xy
        
        # ------------------- 基础速度设置 -------------------
        self.root_states[env_ids, 7:13] = torch.rand(len(env_ids), 6, device=self.device) * 1.0 - 0.5
        
        # ------------------- 更新到仿真 -------------------
        actor_ids = (env_ids * int(self.actors_per_env)).to(dtype=torch.long)
        actor_ids_int32 = actor_ids.to(dtype=torch.int32)
        
        self._actor_root_states[actor_ids] = self.root_states[env_ids]
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._actor_root_states),
            gymtorch.unwrap_tensor(actor_ids_int32),
            len(actor_ids_int32)
        )
    
    def _update_terrain_curriculum(self, env_ids):
        """
        更新地形课程难度
        
        🔧 调整晋级/降级条件，防止难度增长过快
        🚫 跳过指定难度
        
        Args:
            env_ids (List[int]): 需要重置的环境ID
        """
        if not self.init_done:
            return
        
        import torch
        
        # 获取跳过的难度列表
        skip_levels = getattr(self.cfg.terrain, 'skip_terrain_levels', [])
        
        # 计算每个环境走过的距离
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        
        # 晋级条件：走到地形长度的 70% 以上 + tracking reward 足够好
        move_up_distance = distance > (self.terrain.env_length * 0.7)
        
        if hasattr(self, 'episode_sums') and 'tracking_lin_vel' in self.episode_sums:
            tracking_quality = (self.episode_sums["tracking_lin_vel"][env_ids] / 
                              torch.clamp(self.episode_length_buf[env_ids], min=1).float())
            target_reward = 0.4 * self.reward_scales.get("tracking_lin_vel", 1.0)
            move_up = move_up_distance & (tracking_quality > target_reward)
        else:
            move_up = move_up_distance
        
        # 降级条件：走不到地形长度的 30%
        move_down = (distance < self.terrain.env_length * 0.3) & (~move_up)
        
        # 更新地形难度
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        
        # 跳过指定难度
        if len(skip_levels) > 0:
            for env_idx in env_ids:
                current_level = self.terrain_levels[env_idx].item()
                
                if current_level in skip_levels:
                    if move_up[env_ids == env_idx].any():
                        next_level = current_level + 1
                        while next_level in skip_levels and next_level < self.max_terrain_level:
                            next_level += 1
                        self.terrain_levels[env_idx] = min(next_level, self.max_terrain_level - 1)
                    elif move_down[env_ids == env_idx].any():
                        prev_level = current_level - 1
                        while prev_level in skip_levels and prev_level >= 0:
                            prev_level -= 1
                        self.terrain_levels[env_idx] = max(prev_level, 0)
        
        # 达到最高难度的机器人随机分配到中等难度
        max_level = self.max_terrain_level
        max_level_mask = self.terrain_levels[env_ids] >= max_level
        
        if max_level_mask.any():
            available_levels = [i for i in range(max_level // 2, max_level) if i not in skip_levels]
            if len(available_levels) > 0:
                for i, env_idx in enumerate(env_ids[max_level_mask]):
                    random_level = available_levels[torch.randint(0, len(available_levels), (1,)).item()]
                    self.terrain_levels[env_idx] = random_level
            else:
                self.terrain_levels[env_ids[max_level_mask]] = torch.clip(
                    self.terrain_levels[env_ids[max_level_mask]], 0, max_level - 1
                )
        
        # 确保不超出范围
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0, max_level - 1)
        
        # 更新环境原点
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], 
            self.terrain_types[env_ids]
        ]
        
        # 调试输出
        if self.common_step_counter % 1000 == 0:
            avg_level = self.terrain_levels.float().mean().item()
            max_current = self.terrain_levels.max().item()
            print(f"[Terrain Curriculum] Avg level: {avg_level:.2f}, Max: {max_current}/{max_level}")
            if len(skip_levels) > 0:
                print(f"  🚫 Skipping levels: {skip_levels}")
    
    def update_command_curriculum(self, env_ids):
        """
        更新命令速度课程
        
        🔧 修复：正确计算平均 tracking reward，使用实际 episode 长度
        
        ⚠️ 注意：这个方法在 reset_idx 中被调用，但只在 common_step_counter % max_episode_length == 0 时触发
        因此我们应该计算**所有环境**的平均表现，而不仅仅是 env_ids 中的环境
        
        Args:
            env_ids (List[int]): 当前正在重置的环境ID（仅用于触发时机判断）
        """
        if not hasattr(self, 'episode_sums') or 'tracking_lin_vel' not in self.episode_sums:
            return
        
        import numpy as np
        import torch
        
        # 🔧 计算所有环境的平均 tracking reward
        # 使用所有环境而不仅仅是 env_ids，因为命令范围是全局共享的
        all_episode_lengths = torch.clamp(self.episode_length_buf.float(), min=1.0)
        
        # 方法1：使用当前所有环境的平均奖励（包括正在进行的episode）
        current_avg_per_step = torch.mean(self.episode_sums["tracking_lin_vel"] / all_episode_lengths)
        
        # 方法2：使用最近完成的episodes的平均奖励（更稳定）
        # 只考虑episode长度较长的环境（避免刚重置的环境影响统计）
        valid_envs = self.episode_length_buf > (self.max_episode_length * 0.5)
        if valid_envs.any():
            valid_lengths = torch.clamp(self.episode_length_buf[valid_envs].float(), min=1.0)
            avg_tracking_per_step = torch.mean(
                self.episode_sums["tracking_lin_vel"][valid_envs] / valid_lengths
            )
        else:
            avg_tracking_per_step = current_avg_per_step
        
        # 🎯 目标：tracking reward 的平均值达到阈值
        # tracking reward 范围是 [0, reward_scale]（因为 exp(-error^2) 范围是 [0, 1]）
        # curriculum_threshold 表示达到最大奖励的百分比
        target_reward = getattr(self.cfg.commands, 'curriculum_threshold', 0.8) * self.reward_scales["tracking_lin_vel"]
        
        # 🐛 打印调试信息（每次调用时都打印，因为调用频率已经很低了）
        print(f"\n{'='*70}")
        print(f"[Command Curriculum Check] Step {self.common_step_counter}")
        print(f"{'='*70}")
        print(f"  Valid envs: {valid_envs.sum().item()}/{self.num_envs}")
        print(f"  Avg tracking reward (per step): {avg_tracking_per_step:.3f}")
        print(f"  Target reward (80% of {self.reward_scales['tracking_lin_vel']:.1f}): {target_reward:.3f}")
        print(f"  Progress: {(avg_tracking_per_step/target_reward*100):.1f}%")
        print(f"  Current lin_vel_x range: [{self.command_ranges['lin_vel_x'][0]:.2f}, {self.command_ranges['lin_vel_x'][1]:.2f}]")
        print(f"  Max possible range: [{-self.cfg.commands.max_reverse_curriculum:.2f}, {self.cfg.commands.max_curriculum:.2f}]")
        print(f"  Decision: {'✅ EXPAND' if avg_tracking_per_step > target_reward else '⏳ WAIT'}")
        print(f"{'='*70}\n")
        
        # 如果达标，扩展命令范围
        if avg_tracking_per_step > target_reward:
            step = getattr(self.cfg.commands, 'curriculum_step', 0.1)
            
            # 扩展前进速度
            old_max = self.command_ranges["lin_vel_x"][1]
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + step,
                0.,
                self.cfg.commands.max_curriculum
            )
            
            # 扩展后退速度
            old_min = self.command_ranges["lin_vel_x"][0]
            if hasattr(self.cfg.commands, 'max_reverse_curriculum'):
                self.command_ranges["lin_vel_x"][0] = np.clip(
                    self.command_ranges["lin_vel_x"][0] - step * 0.5,
                    -self.cfg.commands.max_reverse_curriculum,
                    0.
                )
            
            # 🐛 调试：打印课程进度（仅当范围真的改变时）
            if old_max != self.command_ranges["lin_vel_x"][1] or old_min != self.command_ranges["lin_vel_x"][0]:
                print(f"\n{'='*60}")
                print(f"[Command Curriculum] ✅✅✅ EXPANDED! ✅✅✅")
                print(f"{'='*60}")
                print(f"  Old range: [{old_min:.2f}, {old_max:.2f}]")
                print(f"  New range: [{self.command_ranges['lin_vel_x'][0]:.2f}, {self.command_ranges['lin_vel_x'][1]:.2f}]")
                print(f"  Avg tracking reward: {avg_tracking_per_step:.3f} (target: {target_reward:.3f})")
                print(f"  Progress: {(avg_tracking_per_step/target_reward*100):.1f}%")
                print(f"{'='*60}\n")
