# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Extended for Imitation Learning (IL) with PPO

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage
from rsl_rl.algorithms.ppo import PPO


class PPOIL(PPO):
    """
    PPO with Imitation Learning (IL)
    
    在标准PPO基础上添加模仿学习损失:
    Total Loss = RL Loss + imitation_coef * IL Loss
    
    IL Loss = MSE(student_actions, teacher_actions)
    
    支持IL系数课程（逐步降低IL权重）:
    - 初期: 高IL权重，快速学习教师行为
    - 后期: 降低IL权重，允许RL优化
    """
    
    def __init__(self,
                 actor_critic,
                 teacher_actor_critic=None,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 # IL specific parameters
                 use_imitation_loss=False,
                 imitation_loss_coef=1.0,
                 imitation_curriculum=False,
                 imitation_coef_schedule=None,
                 ):
        
        # 调用父类初始化
        super().__init__(
            actor_critic=actor_critic,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
        )
        
        # IL specific components
        self.use_imitation_loss = use_imitation_loss
        self.imitation_loss_coef = imitation_loss_coef
        self.imitation_curriculum = imitation_curriculum
        self.imitation_coef_schedule = imitation_coef_schedule or {
            'start': 1.0,
            'end': 0.3,
            'iterations': 1500,
        }
        
        # Teacher model
        self.teacher_actor_critic = teacher_actor_critic
        if self.teacher_actor_critic is not None:
            self.teacher_actor_critic.to(self.device)
            self.teacher_actor_critic.eval()  # 始终在eval模式
            # 冻结教师模型参数
            for param in self.teacher_actor_critic.parameters():
                param.requires_grad = False
        
        # IL metrics
        self.current_imitation_coef = imitation_loss_coef
        self.iteration = 0
        
        print(f"\n🎓 [PPOIL] Imitation Learning initialized")
        print(f"  Use IL loss: {use_imitation_loss}")
        if use_imitation_loss:
            print(f"  IL coef: {imitation_loss_coef}")
            print(f"  IL curriculum: {imitation_curriculum}")
            if imitation_curriculum:
                print(f"  Schedule: {self.imitation_coef_schedule['start']:.2f} → {self.imitation_coef_schedule['end']:.2f} over {self.imitation_coef_schedule['iterations']} iters")
            print(f"  Teacher model: {'Loaded' if teacher_actor_critic else 'None'}")
    
    def update_imitation_coef(self):
        """更新IL系数（根据课程设置）"""
        if not self.imitation_curriculum:
            return self.imitation_loss_coef
        
        schedule = self.imitation_coef_schedule
        start = schedule['start']
        end = schedule['end']
        iterations = schedule['iterations']
        
        if self.iteration >= iterations:
            self.current_imitation_coef = end
        else:
            # 线性插值
            alpha = self.iteration / iterations
            self.current_imitation_coef = start + (end - start) * alpha
        
        return self.current_imitation_coef
    
    def compute_imitation_loss(self, student_obs, student_depth_obs, teacher_obs):
        """
        计算模仿损失
        
        Args:
            student_obs: 学生本体观测 (batch, 45)
            student_depth_obs: 学生深度图像 (batch, 1, H, W)
            teacher_obs: 教师观测 (batch, 45) - 只有本体感知
        
        Returns:
            il_loss: MSE(student_actions, teacher_actions)
        """
        if self.teacher_actor_critic is None:
            return torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():
            # 教师推理（只需要本体观测）
            teacher_actions = self.teacher_actor_critic.act_inference(teacher_obs)
        
        # 学生推理
        from rsl_rl.modules.vision_actor_critic import VisionProprioceptionActorCritic
        is_vision_policy = isinstance(self.actor_critic, VisionProprioceptionActorCritic)
        
        if is_vision_policy and student_depth_obs is not None:
            student_actions = self.actor_critic.act_inference(student_obs, student_depth_obs)
        else:
            student_actions = self.actor_critic.act_inference(student_obs)
        
        # 计算MSE损失
        il_loss = nn.functional.mse_loss(student_actions, teacher_actions)
        
        return il_loss
    
    def update(self):
        """
        PPO + IL更新
        
        在标准PPO更新基础上添加IL损失
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_imitation_loss = 0
        
        # 更新IL系数
        self.update_imitation_coef()
        
        # 检查是否是视觉策略
        from rsl_rl.modules.vision_actor_critic import VisionProprioceptionActorCritic
        is_vision_policy = isinstance(self.actor_critic, VisionProprioceptionActorCritic)
        
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, depth_images_batch in generator:

                # ========== RL Loss (Standard PPO) ==========
                
                # 计算新的动作分布
                if is_vision_policy and depth_images_batch is not None:
                    # 视觉策略: 传递深度图像
                    self.actor_critic.act(obs_batch, depth_images_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    value_batch = self.actor_critic.evaluate(critic_obs_batch, depth_images_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                else:
                    # 标准策略
                    self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                    actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                    value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # RL loss
                rl_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                
                # ========== IL Loss ==========
                imitation_loss = torch.tensor(0.0, device=self.device)
                if self.use_imitation_loss and self.teacher_actor_critic is not None:
                    # 教师只使用本体观测（obs_batch中的45维）
                    teacher_obs = obs_batch[:, :45]  # 假设前45维是本体感知
                    imitation_loss = self.compute_imitation_loss(obs_batch, depth_images_batch, teacher_obs)
                
                # ========== Total Loss ==========
                loss = rl_loss + self.current_imitation_coef * imitation_loss

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_imitation_loss += imitation_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates
        self.storage.clear()
        
        # 增加迭代计数
        self.iteration += 1

        return mean_value_loss, mean_surrogate_loss, mean_imitation_loss
