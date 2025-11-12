#!/usr/bin/env python3
"""
测试 FiLM 门控模块的集成

验证：
1. 配置参数正确读取
2. 网络结构正确构建
3. 前向传播维度正确
4. 初始化时 scale 和 shift 接近 0
"""

import torch
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'rsl_rl'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg
from rsl_rl.modules.vision_actor_critic import VisionProprioceptionActorCritic


def test_film_configuration():
    """测试配置参数"""
    print("=" * 60)
    print("测试 1: FiLM 配置参数")
    print("=" * 60)
    
    cfg = SiriusSharedPPOCfg()
    
    # 检查配置参数是否存在
    assert hasattr(cfg.policy, 'use_film_gating'), "缺少 use_film_gating 配置"
    assert hasattr(cfg.policy, 'film_hidden_dims'), "缺少 film_hidden_dims 配置"
    assert hasattr(cfg.policy, 'film_activation'), "缺少 film_activation 配置"
    assert hasattr(cfg.policy, 'film_scale_limit'), "缺少 film_scale_limit 配置"
    
    print(f"✓ use_film_gating: {cfg.policy.use_film_gating}")
    print(f"✓ film_hidden_dims: {cfg.policy.film_hidden_dims}")
    print(f"✓ film_activation: {cfg.policy.film_activation}")
    print(f"✓ film_scale_limit: {cfg.policy.film_scale_limit}")
    print(f"✓ film_scale_init: {cfg.policy.film_scale_init}")
    print(f"✓ film_shift_init: {cfg.policy.film_shift_init}")
    print("\n配置测试通过！\n")


def test_film_network_structure():
    """测试网络结构"""
    print("=" * 60)
    print("测试 2: FiLM 网络结构")
    print("=" * 60)
    
    # 创建配置 mock
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
    
    # 创建网络（启用 FiLM）
    print("\n创建网络（FiLM 启用）...")
    policy_with_film = VisionProprioceptionActorCritic(
        num_proprio_obs=45,
        num_vision_latent=32,
        num_actions=12,
        vision_encoder_cfg=MockVisionEncoderCfg(),
        use_film_gating=True,
        film_hidden_dims=[64],
        film_activation='elu',
        film_scale_limit=0.1
    )
    
    # 检查 FiLM MLP 是否存在
    assert policy_with_film.use_film, "FiLM 应该被启用"
    assert policy_with_film.film_mlp is not None, "film_mlp 应该存在"
    print(f"✓ FiLM MLP 已创建")
    
    # 检查网络结构
    print(f"\nFiLM MLP 结构:")
    for i, layer in enumerate(policy_with_film.film_mlp):
        print(f"  Layer {i}: {layer}")
    
    # 统计参数
    film_params = sum(p.numel() for p in policy_with_film.film_mlp.parameters())
    total_params = sum(p.numel() for p in policy_with_film.parameters())
    print(f"\n✓ FiLM MLP 参数量: {film_params:,}")
    print(f"✓ 总参数量: {total_params:,}")
    print(f"✓ FiLM 参数占比: {film_params / total_params * 100:.2f}%")
    
    # 创建网络（禁用 FiLM）
    print("\n创建网络（FiLM 禁用）...")
    policy_without_film = VisionProprioceptionActorCritic(
        num_proprio_obs=45,
        num_vision_latent=32,
        num_actions=12,
        vision_encoder_cfg=MockVisionEncoderCfg(),
        use_film_gating=False
    )
    
    assert not policy_without_film.use_film, "FiLM 应该被禁用"
    assert policy_without_film.film_mlp is None, "film_mlp 不应该存在"
    print(f"✓ FiLM 正确禁用\n")
    
    print("网络结构测试通过！\n")
    
    return policy_with_film, policy_without_film


def test_film_forward_pass(policy_with_film, policy_without_film):
    """测试前向传播"""
    print("=" * 60)
    print("测试 3: 前向传播")
    print("=" * 60)
    
    batch_size = 16
    proprio_obs = torch.rand(batch_size, 45)
    depth_image = torch.rand(batch_size, 1, 58, 87)
    
    print(f"输入维度:")
    print(f"  proprio_obs: {proprio_obs.shape}")
    print(f"  depth_image: {depth_image.shape}")
    
    # 测试 FiLM 启用的情况
    print("\n测试 FiLM 启用的前向传播...")
    with torch.no_grad():
        # 测试融合
        fused_with_film = policy_with_film._fuse_observations(proprio_obs, depth_image)
        print(f"✓ 融合后维度: {fused_with_film.shape}")
        assert fused_with_film.shape == (batch_size, 77), f"融合后维度错误: {fused_with_film.shape}"
        
        # 测试 act
        actions = policy_with_film.act(proprio_obs, depth_image)
        print(f"✓ 动作维度: {actions.shape}")
        assert actions.shape == (batch_size, 12), f"动作维度错误: {actions.shape}"
        
        # 测试 evaluate
        values = policy_with_film.evaluate(proprio_obs, depth_image)
        print(f"✓ 值函数维度: {values.shape}")
        assert values.shape == (batch_size, 1), f"值函数维度错误: {values.shape}"
    
    # 测试 FiLM 禁用的情况
    print("\n测试 FiLM 禁用的前向传播...")
    with torch.no_grad():
        fused_without_film = policy_without_film._fuse_observations(proprio_obs, depth_image)
        print(f"✓ 融合后维度: {fused_without_film.shape}")
        assert fused_without_film.shape == (batch_size, 77), f"融合后维度错误: {fused_without_film.shape}"
    
    print("\n前向传播测试通过！\n")


def test_film_initialization(policy_with_film):
    """测试 FiLM 初始化"""
    print("=" * 60)
    print("测试 4: FiLM 初始化（scale 和 shift 应接近 0）")
    print("=" * 60)
    
    batch_size = 16
    proprio_obs = torch.rand(batch_size, 45)
    depth_image = torch.rand(batch_size, 1, 58, 87)
    
    with torch.no_grad():
        # 获取视觉特征
        vision_latent = policy_with_film.vision_encoder(depth_image)
        
        # 通过 FiLM MLP
        film_out = policy_with_film.film_mlp(vision_latent)
        scale, shift = torch.split(film_out, 45, dim=-1)
        
        # 应用限幅
        scale = policy_with_film.film_scale_limit * torch.tanh(scale)
        
        print(f"FiLM 输出统计（初始化后）:")
        print(f"  scale 均值: {scale.mean().item():.6f}")
        print(f"  scale 标准差: {scale.std().item():.6f}")
        print(f"  scale 范围: [{scale.min().item():.6f}, {scale.max().item():.6f}]")
        print(f"  shift 均值: {shift.mean().item():.6f}")
        print(f"  shift 标准差: {shift.std().item():.6f}")
        print(f"  shift 范围: [{shift.min().item():.6f}, {shift.max().item():.6f}]")
        
        # 检查是否接近 0（零初始化的效果）
        assert abs(scale.mean().item()) < 0.01, "scale 均值应接近 0"
        assert abs(shift.mean().item()) < 0.01, "shift 均值应接近 0"
        
        # 测试调制效果
        modulated = proprio_obs * (1.0 + scale) + shift
        diff = (modulated - proprio_obs).abs().mean()
        print(f"\n调制效果:")
        print(f"  |modulated - proprio| 平均差异: {diff.item():.6f}")
        print(f"  相对差异: {(diff / proprio_obs.abs().mean()).item() * 100:.2f}%")
        
        assert diff < 0.1, "初期调制应该很小（接近恒等变换）"
        
        print(f"\n✓ 初始化正确：scale 和 shift 接近 0，退化为恒等变换")
    
    print("\nFiLM 初始化测试通过！\n")


def main():
    print("\n" + "=" * 60)
    print("FiLM 门控模块集成测试")
    print("=" * 60 + "\n")
    
    try:
        # 测试 1: 配置
        test_film_configuration()
        
        # 测试 2: 网络结构
        policy_with_film, policy_without_film = test_film_network_structure()
        
        # 测试 3: 前向传播
        test_film_forward_pass(policy_with_film, policy_without_film)
        
        # 测试 4: 初始化
        test_film_initialization(policy_with_film)
        
        print("=" * 60)
        print("✅ 所有测试通过！FiLM 模块集成成功！")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
