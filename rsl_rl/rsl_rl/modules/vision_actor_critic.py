# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
视觉-本体融合的 Actor-Critic 网络

将深度图像编码后与本体感觉观测融合，用于端到端视觉强化学习。
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import ActorCritic, get_activation
from .vision_encoder import SimpleCNNEncoder


class VisionProprioceptionActorCritic(ActorCritic):
    """
    融合视觉和本体感觉的 Actor-Critic 网络。
    
    架构:
        深度图 (B, 1, H, W) 
            ↓
        CNN编码器
            ↓
        视觉特征 (B, vision_latent_dim)
            ↓ (concatenate)
        本体观测 (B, proprio_dim) → 融合特征 (B, proprio_dim + vision_latent_dim)
            ↓
        Actor/Critic MLP
            ↓
        动作/值函数
    
    Args:
        num_proprio_obs: 本体感觉观测维度 (例如 45)
        num_vision_latent: 视觉特征维度 (例如 32)
        num_actions: 动作维度
        vision_encoder_cfg: 视觉编码器配置
        actor_hidden_dims: Actor MLP 隐藏层维度
        critic_hidden_dims: Critic MLP 隐藏层维度
        activation: 激活函数
        init_noise_std: 初始动作噪声标准差
    """
    
    def __init__(self,
                 num_proprio_obs,
                 num_vision_latent,
                 num_actions,
                 vision_encoder_cfg,
                 actor_hidden_dims=[256, 128, 64],
                 critic_hidden_dims=[256, 128, 64],
                 activation='elu',
                 init_noise_std=1.0,
                 # 🎬 FiLM 门控参数（从 policy 配置传入）
                 use_film_gating=True,
                 film_hidden_dims=[64],
                 film_activation='elu',
                 film_scale_init=0.0,
                 film_shift_init=0.0,
                 film_scale_limit=0.1,
                 **kwargs):
        
        # 存储维度信息
        self.num_proprio_obs = num_proprio_obs
        self.num_vision_latent = num_vision_latent
        self.num_fused_obs = num_proprio_obs + num_vision_latent
        
        # 初始化父类（使用融合后的观测维度）
        super(VisionProprioceptionActorCritic, self).__init__(
            num_actor_obs=self.num_fused_obs,
            num_critic_obs=self.num_fused_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            **kwargs
        )
        
        # 构建视觉编码器
        self.vision_encoder = SimpleCNNEncoder(
            input_height=vision_encoder_cfg.input_height,
            input_width=vision_encoder_cfg.input_width,
            input_channels=vision_encoder_cfg.input_channels,
            latent_dim=vision_encoder_cfg.latent_dim,
            cnn_layers=vision_encoder_cfg.cnn_layers,
            activation=vision_encoder_cfg.activation,
            use_batch_norm=vision_encoder_cfg.use_batch_norm,
            dropout=vision_encoder_cfg.dropout
        )
        
        # 🎬 构建 FiLM 门控网络（Feature-wise Linear Modulation）
        # FiLM 参数直接从函数参数获取（由 build_vision_actor_critic 从 cfg.policy 传入）
        self.use_film = use_film_gating
        self.film_scale_limit = film_scale_limit
        
        if self.use_film:
            # 获取激活函数
            film_act = get_activation(film_activation)
            
            # 构建门控 MLP: vision_latent (32) -> [64] -> scale (45) + shift (45)
            # 输出维度是 2 * num_proprio_obs，前半部分是 scale，后半部分是 shift
            dims = [self.num_vision_latent] + film_hidden_dims + [2 * self.num_proprio_obs]
            layers = []
            
            # 隐藏层：添加线性层 + 激活函数
            for i in range(len(dims) - 2):
                layers += [nn.Linear(dims[i], dims[i+1]), film_act]
            
            # 输出层：只有线性层，不加激活函数（因为 scale 和 shift 需要不同的处理）
            layers += [nn.Linear(dims[-2], dims[-1])]
            
            self.film_mlp = nn.Sequential(*layers)
            
            # 🔧 关键：零初始化输出层，确保训练初期是恒等映射
            # 这样初期 scale ≈ 0, shift ≈ 0，modulated_proprio ≈ proprio
            # 网络可以从"纯 concat"的稳定状态开始，逐步学习门控
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)
            
            print(f"\n[FiLM Gating Module] Enabled")
            print(f"  Input dim: {self.num_vision_latent} (vision latent)")
            print(f"  Hidden dims: {film_hidden_dims}")
            print(f"  Output dim: {2 * self.num_proprio_obs} (scale + shift)")
            print(f"  Scale limit: ±{self.film_scale_limit} (via tanh)")
            print(f"  Activation: {film_activation}")
        else:
            self.film_mlp = None
            print(f"\n[FiLM Gating Module] Disabled (using simple concatenation)")
        
        print(f"\n[VisionProprioceptionActorCritic]")
        print(f"  Proprio obs dim: {num_proprio_obs}")
        print(f"  Vision latent dim: {num_vision_latent}")
        print(f"  Fused obs dim: {self.num_fused_obs}")
        print(f"  Vision encoder info: {self.vision_encoder.get_output_info()}")
    
    def _fuse_observations(self, proprio_obs, depth_image):
        """
        融合本体观测和视觉特征
        
        使用 FiLM (Feature-wise Linear Modulation) 门控机制：
        1. 视觉编码器提取视觉特征
        2. 门控 MLP 从视觉特征生成 scale 和 shift
        3. 使用 scale 和 shift 调制本体特征：modulated = proprio * (1 + scale) + shift
        4. 拼接调制后的本体特征和视觉特征
        
        Args:
            proprio_obs: (B, num_proprio_obs) 本体感觉观测
            depth_image: (B, 1, H, W) 归一化深度图
        
        Returns:
            fused_obs: (B, num_fused_obs) 融合后的观测
        """
        # 编码视觉输入
        vision_latent = self.vision_encoder(depth_image)  # (B, num_vision_latent)
        
        if self.use_film:
            # 🎬 FiLM 门控调制
            # 1. 使用门控 MLP 从视觉特征生成 scale 和 shift
            film_out = self.film_mlp(vision_latent)  # (B, 2 * num_proprio_obs)
            scale, shift = torch.split(film_out, self.num_proprio_obs, dim=-1)  # (B, P), (B, P)
            
            # 2. 使用 tanh 将 scale 限制在 ±film_scale_limit 范围内
            #    这样可以避免训练初期的梯度爆炸，保证数值稳定性
            #    例如：film_scale_limit=0.1，则 scale ∈ [-0.1, 0.1]
            scale = self.film_scale_limit * torch.tanh(scale)
            
            # 3. FiLM 调制公式：modulated = proprio * (1 + scale) + shift
            #    - (1 + scale)：乘性调制，初期 scale≈0 时退化为恒等变换
            #    - shift：加性调制，提供偏移能力
            modulated_proprio = proprio_obs * (1.0 + scale) + shift
        else:
            # 如果禁用 FiLM，直接使用原始本体观测（向后兼容）
            modulated_proprio = proprio_obs
        
        # 拼接调制后的本体特征和视觉特征
        fused_obs = torch.cat([modulated_proprio, vision_latent], dim=-1)  # (B, num_fused_obs)
        
        return fused_obs
    
    def update_distribution(self, proprio_obs, depth_image):
        """
        更新动作分布
        
        Args:
            proprio_obs: (B, num_proprio_obs) 本体感觉观测
            depth_image: (B, 1, H, W) 深度图
        """
        # 融合观测
        fused_obs = self._fuse_observations(proprio_obs, depth_image)
        
        # 通过 Actor 得到动作均值
        mean = self.actor(fused_obs)
        self.distribution = Normal(mean, mean * 0. + self.std)
    
    def act(self, proprio_obs, depth_image, masks=None, hidden_states=None):
        """
        采样动作
        
        Args:
            proprio_obs: (B, num_proprio_obs)
            depth_image: (B, 1, H, W)
            masks: (可选) 用于RNN，这里忽略
            hidden_states: (可选) 用于RNN，这里忽略
        
        Returns:
            actions: (B, num_actions)
        
        设计说明:
        - 接受masks和hidden_states参数以兼容PPO的调用接口
        - 但实际不使用（因为我们不是RNN）
        """
        self.update_distribution(proprio_obs, depth_image)
        return self.distribution.sample()
    
    def act_inference(self, proprio_obs, depth_image):
        """
        推理模式（确定性动作）
        
        Args:
            proprio_obs: (B, num_proprio_obs)
            depth_image: (B, 1, H, W)
        
        Returns:
            actions_mean: (B, num_actions)
        """
        fused_obs = self._fuse_observations(proprio_obs, depth_image)
        actions_mean = self.actor(fused_obs)
        return actions_mean
    
    def evaluate(self, proprio_obs, depth_image, masks=None, hidden_states=None):
        """
        评估值函数
        
        Args:
            proprio_obs: (B, num_proprio_obs)
            depth_image: (B, 1, H, W)
            masks: (可选) 用于RNN，这里忽略
            hidden_states: (可选) 用于RNN，这里忽略
        
        Returns:
            value: (B, 1)
        
        设计说明:
        - 接受masks和hidden_states参数以兼容PPO的调用接口
        - 但实际不使用（因为我们不是RNN）
        """
        fused_obs = self._fuse_observations(proprio_obs, depth_image)
        value = self.critic(fused_obs)
        return value


def build_vision_actor_critic(cfg, num_actions):
    """
    从配置构建视觉-本体融合的 Actor-Critic
    
    Args:
        cfg: 包含 policy 和 vision_encoder 的配置对象
        num_actions: 动作维度
    
    Returns:
        VisionProprioceptionActorCritic 实例
    
    Example:
        >>> from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg
        >>> cfg = SiriusSharedPPOCfg()
        >>> policy = build_vision_actor_critic(cfg, num_actions=12)
    """
    # 计算本体观测维度: 总观测维度 - 视觉特征维度
    # 例如: 77 - 32 = 45
    num_proprio_obs = cfg.env.num_observations - cfg.policy.vision_latent_dim
    
    actor_critic = VisionProprioceptionActorCritic(
        num_proprio_obs=num_proprio_obs,
        num_vision_latent=cfg.policy.vision_latent_dim,
        num_actions=num_actions,
        vision_encoder_cfg=cfg.vision_encoder,
        actor_hidden_dims=cfg.policy.actor_hidden_dims,
        critic_hidden_dims=cfg.policy.critic_hidden_dims,
        activation=cfg.policy.activation,
        init_noise_std=cfg.policy.init_noise_std if hasattr(cfg.policy, 'init_noise_std') else 1.0,
        # 🎬 传递 FiLM 门控参数（从 cfg.policy 读取，提供默认值以保证向后兼容）
        use_film_gating=getattr(cfg.policy, 'use_film_gating', True),
        film_hidden_dims=getattr(cfg.policy, 'film_hidden_dims', [64]),
        film_activation=getattr(cfg.policy, 'film_activation', 'elu'),
        film_scale_init=getattr(cfg.policy, 'film_scale_init', 0.0),
        film_shift_init=getattr(cfg.policy, 'film_shift_init', 0.0),
        film_scale_limit=getattr(cfg.policy, 'film_scale_limit', 0.1),
    )
    
    return actor_critic


# 测试代码
if __name__ == "__main__":
    print("Testing VisionProprioceptionActorCritic...")
    
    # 创建一个简单的配置mock
    class MockVisionEncoderCfg:
        input_height = 58
        input_width = 87
        input_channels = 1
        latent_dim = 32
        cnn_layers = [
            {'out_channels': 16, 'kernel_size': 5, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'use_maxpool': False},
        ]
        activation = 'relu'
        use_batch_norm = False
        dropout = 0.0
    
    # 创建网络
    policy = VisionProprioceptionActorCritic(
        num_proprio_obs=45,
        num_vision_latent=32,
        num_actions=12,
        vision_encoder_cfg=MockVisionEncoderCfg(),
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation='elu'
    )
    
    print(f"\nPolicy network:")
    print(f"  Total parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # 测试前向传播
    batch_size = 16
    proprio_obs = torch.rand(batch_size, 45)
    depth_image = torch.rand(batch_size, 1, 58, 87)
    
    print(f"\nTesting forward pass...")
    print(f"  Proprio obs shape: {proprio_obs.shape}")
    print(f"  Depth image shape: {depth_image.shape}")
    
    # 测试act
    actions = policy.act(proprio_obs, depth_image)
    print(f"  Actions shape: {actions.shape}")
    
    # 测试evaluate
    values = policy.evaluate(proprio_obs, depth_image)
    print(f"  Values shape: {values.shape}")
    
    # 测试act_inference
    actions_mean = policy.act_inference(proprio_obs, depth_image)
    print(f"  Actions mean shape: {actions_mean.shape}")
    
    print("\n✓ VisionProprioceptionActorCritic test passed!")
