#!/usr/bin/env python3
"""
简化的视觉RL模块测试 - 不依赖 Isaac Gym

只测试纯PyTorch组件:
- SimpleCNNEncoder
- VisionProprioceptionActorCritic
"""

import os
import sys

# Add paths  
LEGGED_GYM_ROOT = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(os.path.join(LEGGED_GYM_ROOT, 'rsl_rl'))

import torch

print("=" * 80)
print("视觉RL模块单元测试 (不含Isaac Gym)")
print("=" * 80)

# ====================================================================
# 测试 1: SimpleCNNEncoder
# ====================================================================
print("\n[测试 1/3] SimpleCNNEncoder...")
try:
    from rsl_rl.modules.vision_encoder import SimpleCNNEncoder
    
    # 创建编码器
    encoder = SimpleCNNEncoder(
        input_height=58,
        input_width=87,
        input_channels=1,
        latent_dim=32,
        cnn_layers=[
            {'out_channels': 16, 'kernel_size': 5, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
            {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'use_maxpool': False},
        ],
        activation='relu',
        use_batch_norm=False,
        dropout=0.0
    )
    
    # 测试
    batch_size = 16
    depth_input = torch.rand(batch_size, 1, 58, 87)
    latent_output = encoder(depth_input)
    
    assert latent_output.shape == (batch_size, 32), f"Wrong output shape: {latent_output.shape}"
    
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ SimpleCNNEncoder 测试通过")
    print(f"  输入: {depth_input.shape} → 输出: {latent_output.shape}")
    print(f"  参数数量: {n_params:,}")
    
except Exception as e:
    print(f"✗ SimpleCNNEncoder 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 测试 2: VisionProprioceptionActorCritic
# ====================================================================
print("\n[测试 2/3] VisionProprioceptionActorCritic...")
try:
    from rsl_rl.modules.vision_actor_critic import VisionProprioceptionActorCritic
    
    # 创建mock vision_encoder_cfg
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
    
    # 创建policy
    policy = VisionProprioceptionActorCritic(
        num_proprio_obs=45,
        num_vision_latent=32,
        num_actions=12,
        vision_encoder_cfg=MockVisionEncoderCfg(),
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation='elu'
    )
    
    # 测试前向传播
    batch_size = 16
    proprio_input = torch.rand(batch_size, 45)
    depth_input = torch.rand(batch_size, 1, 58, 87)
    
    # act
    actions = policy.act(proprio_input, depth_input)
    assert actions.shape == (batch_size, 12), f"Wrong actions shape: {actions.shape}"
    
    # evaluate
    values = policy.evaluate(proprio_input, depth_input)
    assert values.shape == (batch_size, 1), f"Wrong values shape: {values.shape}"
    
    # act_inference
    actions_mean = policy.act_inference(proprio_input, depth_input)
    assert actions_mean.shape == (batch_size, 12), f"Wrong actions_mean shape: {actions_mean.shape}"
    
    total_params = sum(p.numel() for p in policy.parameters())
    vision_params = sum(p.numel() for p in policy.vision_encoder.parameters())
    
    print(f"✓ VisionProprioceptionActorCritic 测试通过")
    print(f"  输入: proprio{proprio_input.shape} + depth{depth_input.shape}")
    print(f"  输出: actions{actions.shape}, values{values.shape}")
    print(f"  参数数量: {total_params:,} (vision: {vision_params:,})")
    
except Exception as e:
    print(f"✗ VisionProprioceptionActorCritic 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 测试 3: 大批量测试 (模拟实际训练)
# ====================================================================
print("\n[测试 3/3] 大批量模拟测试...")
try:
    num_envs = 1024
    proprio_input = torch.rand(num_envs, 45)
    depth_input = torch.rand(num_envs, 1, 58, 87)
    
    with torch.no_grad():
        actions = policy.act(proprio_input, depth_input)
        values = policy.evaluate(proprio_input, depth_input)
    
    assert actions.shape == (num_envs, 12)
    assert values.shape == (num_envs, 1)
    
    print(f"✓ 大批量测试通过 (num_envs={num_envs})")
    print(f"  actions: {actions.shape}, values: {values.shape}")
    
except Exception as e:
    print(f"✗ 大批量测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 总结
# ====================================================================
print("\n" + "=" * 80)
print("✓ 所有模块测试通过!")
print("=" * 80)
print("\n组件就绪:")
print("  ✓ SimpleCNNEncoder: 深度编码器")
print("  ✓ VisionProprioceptionActorCritic: 视觉-本体融合策略")
print("\n下一步:")
print("  - 修改 PPO 和 Runner 来传递 depth_obs_buf")
print("  - 测试完整训练流程")
print("")
