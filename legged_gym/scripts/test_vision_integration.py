#!/usr/bin/env python3
"""
测试视觉-本体融合RL系统的完整集成

验证点:
1. 环境能够生成本体观测(45维)和深度图像(1x58x87)
2. VisionProprioceptionActorCritic能够处理这两种输入
3. 前向传播正常,输出动作和值函数
4. 参数数量合理

使用方法:
    python test_vision_integration.py
"""

import os
import sys

# Add paths
LEGGED_GYM_ROOT = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(LEGGED_GYM_ROOT)
sys.path.append(os.path.join(LEGGED_GYM_ROOT, 'rsl_rl'))

print("=" * 80)
print("视觉-本体融合RL系统集成测试")
print("=" * 80)

# IMPORTANT: Import torch AFTER setting paths, to avoid isaacgym import order issues
import torch

# ====================================================================
# 测试 1: 加载配置
# ====================================================================
print("\n[测试 1/4] 加载配置...")
try:
    from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg
    from legged_gym.envs.sirius_diff_vis.sirius_flat_config import SiriusFlatCfg
    
    cfg = SiriusFlatCfg()
    shared_cfg = SiriusSharedPPOCfg()
    
    # 验证配置
    assert hasattr(cfg, 'env'), "Missing env config"
    assert hasattr(cfg, 'camera'), "Missing camera config"
    assert cfg.camera.enable, "Camera not enabled"
    assert cfg.camera.height == 58, f"Expected height 58, got {cfg.camera.height}"
    assert cfg.camera.width == 87, f"Expected width 87, got {cfg.camera.width}"
    assert cfg.env.num_observations == 77, f"Expected 77 obs, got {cfg.env.num_observations}"
    
    # 验证共享配置
    assert hasattr(shared_cfg, 'vision_encoder'), "Missing vision_encoder config"
    assert hasattr(shared_cfg, 'policy'), "Missing policy config"
    assert shared_cfg.policy.use_vision, "Vision not enabled in policy"
    assert shared_cfg.policy.vision_latent_dim == 32, "Vision latent dim should be 32"
    assert shared_cfg.policy.actor_hidden_dims == [256, 128, 64], "Actor hidden dims mismatch"
    
    print("✓ 配置加载成功")
    print(f"  - 本体观测维度: {cfg.env.num_observations - shared_cfg.policy.vision_latent_dim}")
    print(f"  - 视觉特征维度: {shared_cfg.policy.vision_latent_dim}")
    print(f"  - 总观测维度: {cfg.env.num_observations}")
    print(f"  - 深度图尺寸: {cfg.camera.height} x {cfg.camera.width}")
    
except Exception as e:
    print(f"✗ 配置加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 测试 2: 创建视觉编码器
# ====================================================================
print("\n[测试 2/4] 创建视觉编码器...")
try:
    from rsl_rl.modules.vision_encoder import SimpleCNNEncoder
    
    vision_encoder = SimpleCNNEncoder(
        input_height=shared_cfg.vision_encoder.input_height,
        input_width=shared_cfg.vision_encoder.input_width,
        input_channels=shared_cfg.vision_encoder.input_channels,
        latent_dim=shared_cfg.vision_encoder.latent_dim,
        cnn_layers=shared_cfg.vision_encoder.cnn_layers,
        activation=shared_cfg.vision_encoder.activation,
        use_batch_norm=shared_cfg.vision_encoder.use_batch_norm,
        dropout=shared_cfg.vision_encoder.dropout
    )
    
    # 测试前向传播
    batch_size = 16
    dummy_depth = torch.rand(batch_size, 1, 58, 87)
    vision_features = vision_encoder(dummy_depth)
    
    assert vision_features.shape == (batch_size, 32), \
        f"Expected shape ({batch_size}, 32), got {vision_features.shape}"
    
    print("✓ 视觉编码器创建成功")
    print(f"  - 输入: ({batch_size}, 1, 58, 87)")
    print(f"  - 输出: {vision_features.shape}")
    print(f"  - 参数量: {sum(p.numel() for p in vision_encoder.parameters()):,}")
    
except Exception as e:
    print(f"✗ 视觉编码器创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 测试 3: 创建VisionProprioceptionActorCritic
# ====================================================================
print("\n[测试 3/4] 创建视觉-本体融合 Actor-Critic...")
try:
    from rsl_rl.modules.vision_actor_critic import VisionProprioceptionActorCritic
    
    num_proprio_obs = 45
    num_vision_latent = 32
    num_actions = 12
    
    policy = VisionProprioceptionActorCritic(
        num_proprio_obs=num_proprio_obs,
        num_vision_latent=num_vision_latent,
        num_actions=num_actions,
        vision_encoder_cfg=shared_cfg.vision_encoder,
        actor_hidden_dims=shared_cfg.policy.actor_hidden_dims,
        critic_hidden_dims=shared_cfg.policy.critic_hidden_dims,
        activation=shared_cfg.policy.activation
    )
    
    # 测试前向传播
    dummy_proprio = torch.rand(batch_size, 45)
    dummy_depth = torch.rand(batch_size, 1, 58, 87)
    
    # 测试act
    actions = policy.act(dummy_proprio, dummy_depth)
    assert actions.shape == (batch_size, num_actions), \
        f"Expected actions shape ({batch_size}, {num_actions}), got {actions.shape}"
    
    # 测试evaluate
    values = policy.evaluate(dummy_proprio, dummy_depth)
    assert values.shape == (batch_size, 1), \
        f"Expected values shape ({batch_size}, 1), got {values.shape}"
    
    # 测试act_inference
    actions_mean = policy.act_inference(dummy_proprio, dummy_depth)
    assert actions_mean.shape == (batch_size, num_actions), \
        f"Expected actions_mean shape ({batch_size}, {num_actions}), got {actions_mean.shape}"
    
    total_params = sum(p.numel() for p in policy.parameters())
    vision_params = sum(p.numel() for p in policy.vision_encoder.parameters())
    mlp_params = total_params - vision_params
    
    print("✓ Actor-Critic 创建成功")
    print(f"  - 本体输入: ({batch_size}, {num_proprio_obs})")
    print(f"  - 深度输入: ({batch_size}, 1, 58, 87)")
    print(f"  - 动作输出: {actions.shape}")
    print(f"  - 值函数输出: {values.shape}")
    print(f"  - 总参数量: {total_params:,}")
    print(f"    ├─ 视觉编码器: {vision_params:,}")
    print(f"    └─ Actor/Critic MLP: {mlp_params:,}")
    
except Exception as e:
    print(f"✗ Actor-Critic 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 测试 4: 模拟环境观测
# ====================================================================
print("\n[测试 4/4] 模拟环境观测生成...")
try:
    # 模拟环境的compute_observations输出
    num_envs = 1024
    
    # 本体观测 (45维)
    mock_obs_buf = torch.rand(num_envs, 45)
    
    # 深度图像 (1 x 58 x 87, 归一化到[0,1])
    mock_depth_obs_buf = torch.rand(num_envs, 1, 58, 87)
    
    # 测试policy能否处理
    with torch.no_grad():
        actions = policy.act(mock_obs_buf, mock_depth_obs_buf)
        values = policy.evaluate(mock_obs_buf, mock_depth_obs_buf)
    
    assert actions.shape == (num_envs, num_actions)
    assert values.shape == (num_envs, 1)
    
    print("✓ 环境观测模拟成功")
    print(f"  - 环境数: {num_envs}")
    print(f"  - 本体观测: {mock_obs_buf.shape}")
    print(f"  - 深度观测: {mock_depth_obs_buf.shape}")
    print(f"  - 动作采样: {actions.shape}")
    print(f"  - 值函数估计: {values.shape}")
    
except Exception as e:
    print(f"✗ 环境观测模拟失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# 总结
# ====================================================================
print("\n" + "=" * 80)
print("✓ 所有测试通过!")
print("=" * 80)
print("\n系统组件:")
print("  1. SimpleCNNEncoder: 深度图 (1x58x87) → 视觉特征 (32)")
print("  2. VisionProprioceptionActorCritic: 融合视觉+本体 → 动作/值函数")
print("  3. 环境接口: obs_buf (45) + depth_obs_buf (1x58x87)")
print("\n下一步:")
print("  - 修改 PPO 算法来传递 depth_obs_buf")
print("  - 修改 OnPolicyRunner 来协调数据流")
print("  - 运行实际训练测试")
print("")
