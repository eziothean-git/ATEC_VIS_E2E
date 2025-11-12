# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
共享的模型配置，确保 task=sirius 和 task=sirius_diff_vis 使用相同的网络结构。
这样可以在两个任务之间使用 --resume 无缝衔接。

当前阶段：保持与现有配置兼容，不引入新功能。
"""

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO


class SiriusSharedPPOCfg:
    """
    Sirius 任务共享的 PPO 模型配置。
    
    设计原则：
    1. policy 配置在两个任务间完全一致，确保网络结构相同
    2. runner 配置允许各任务独立覆盖（实验名、迭代次数等）
    3. 保持与 rsl_rl 的兼容性
    
    视觉-本体融合架构：
    - 深度图 (87x58) -> CNN编码器 -> 32维特征
    - 本体观测 45维 + 视觉特征 32维 = 77维输入
    - 融合后送入 Actor/Critic MLP
    """
    
    class vision_encoder:
        """
        视觉编码器配置 - 3层CNN将深度图编码为低维特征
        
        输入: (B, 1, 58, 87) - 归一化到[0,1]的深度图
        输出: (B, 32) - 视觉特征向量
        
        架构:
        - Conv1: 1 -> 16 channels, kernel=5, stride=2, maxpool(2) -> ~14x21
        - Conv2: 16 -> 32 channels, kernel=3, stride=2, maxpool(2) -> ~3x5  
        - Conv3: 32 -> 32 channels, kernel=3, stride=1, flatten -> 32 dims
        """
        # 输入深度图尺寸
        input_height = 58
        input_width = 87
        input_channels = 1  # 单通道深度图
        
        # CNN 层配置
        # 格式: [out_channels, kernel_size, stride, use_maxpool]
        cnn_layers = [
            {'out_channels': 16, 'kernel_size': 5, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'use_maxpool': False},
        ]
        
        # 输出特征维度
        latent_dim = 32
        
        # 激活函数
        activation = 'relu'  # relu | elu | leaky_relu
        
        # Batch normalization (可选,训练时可能有帮助)
        use_batch_norm = False
        
        # Dropout (可选,防止过拟合)
        dropout = 0.0  # 0.0 表示不使用
    
    class policy(LeggedRobotCfgPPO.policy):
        """
        策略网络配置 - 两个任务必须完全相同！
        
        输入维度:
        - proprio_dim = 45 (本体感觉)
        - vision_dim = 32 (视觉特征)
        - total = 77
        
        当前使用较大的网络容量 [256, 128, 64]，理由：
        - 足够的表达能力处理桥梁等复杂场景
        - 可以从平地迁移到桥梁（简单->复杂）
        - 避免后期因容量不足需要重新训练
        """
        # MLP 结构 (接收 77 维融合输入)
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        
        # 视觉编码器配置（必须一致！）
        use_vision = True  # 标记使用视觉输入
        vision_latent_dim = 32  # 必须与 vision_encoder.latent_dim 一致
        
        # 🎬 FiLM/Gating 配置 - 使用视觉特征调制本体特征
        # FiLM (Feature-wise Linear Modulation): modulated_proprio = proprio * (1 + scale) + shift
        # 这种门控机制让视觉特征能够动态调制本体特征，增强多模态融合能力
        use_film_gating = True  # 是否启用 FiLM 门控（默认启用）
        film_hidden_dims = [64]  # 门控 MLP 隐藏层维度：vision_latent (32) -> [64] -> scale(45) + shift(45)
        film_activation = 'elu'  # 门控 MLP 的激活函数
        
        # FiLM 参数初始化和限制（保证训练稳定性）
        film_scale_init = 0.0   # scale 的初始值（0.0 表示初期接近恒等变换）
        film_shift_init = 0.0   # shift 的初始值（0.0 表示初期无偏移）
        film_scale_limit = 0.1  # scale 的限制范围（通过 tanh 限幅到 ±0.1，避免初期梯度爆炸）
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        """
        PPO 算法超参数 - 两个任务共享
        
        注意：学习率等训练相关参数在 runner 中设置，允许各任务独立调整
        """
        # 使用基类的默认值，如果需要可以在这里覆盖
        # value_loss_coef = 1.0
        # use_clipped_value_loss = True
        # clip_param = 0.2
        # entropy_coef = 0.01
        # num_learning_epochs = 5
        # num_mini_batches = 4
        # learning_rate = 1.e-3  # 不建议在这里设置，应该在 runner 中
        pass
    
    class runner(LeggedRobotCfgPPO.runner):
        """
        训练运行配置的默认值 - 各任务可以独立覆盖
        
        子任务应该覆盖的关键参数：
        - experiment_name: 实验名称（区分任务）
        - max_iterations: 训练迭代次数
        - learning_rate: 学习率（微调时可能需要降低）
        - resume/load_run/checkpoint: 恢复训练相关
        """
        # 使用视觉-本体融合的 Actor-Critic 类
        policy_class_name = 'VisionProprioceptionActorCritic'
        
        # 这里只设置通用默认值，具体任务会覆盖
        run_name = ''
        max_iterations = 1200
        
        # 这些参数各任务必须自己设置
        # experiment_name = "must_be_overridden"
        
        # Resume 相关（用于阶段2从阶段1继续）
        # resume = False
        # load_run = -1
        # checkpoint = -1


# 为了方便导入，提供一个别名
SiriusSharedModelCfg = SiriusSharedPPOCfg
