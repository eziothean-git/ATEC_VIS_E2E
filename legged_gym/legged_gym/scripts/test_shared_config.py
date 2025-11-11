#!/usr/bin/env python3
"""
快速测试：验证共享模型配置是否正确加载

这个脚本不运行训练，只检查配置能否正确导入和实例化。
"""

import sys
import os

# Add legged_gym to path
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LEGGED_GYM_ROOT)

print("测试共享模型配置...")
print()

# Test 1: Import shared config
print("1. 导入共享配置...")
try:
    from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg
    print("   ✓ sirius_shared_model.py 导入成功")
except Exception as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

# Test 2: Import sirius flat config
print("2. 导入 sirius 平地配置...")
try:
    from legged_gym.envs.sirius_diff_vis.sirius_flat_config import SiriusFlatCfg, SiriusFlatCfgPPO
    flat_cfg = SiriusFlatCfg()
    flat_ppo_cfg = SiriusFlatCfgPPO()
    print("   ✓ SiriusFlatCfg 实例化成功")
    print(f"     - Actor dims: {flat_ppo_cfg.policy.actor_hidden_dims}")
    print(f"     - Critic dims: {flat_ppo_cfg.policy.critic_hidden_dims}")
    print(f"     - Activation: {flat_ppo_cfg.policy.activation}")
    print(f"     - Experiment: {flat_ppo_cfg.runner.experiment_name}")
except Exception as e:
    print(f"   ✗ 导入或实例化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Import sirius bridge config
print("3. 导入 sirius_diff_vis 桥梁配置...")
try:
    from legged_gym.envs.sirius_diff_vis.sirius_bridge_env import (
        SiriusTwoSpanBridgeCfg, 
        SiriusTwoSpanBridgeCfgPPO
    )
    bridge_cfg = SiriusTwoSpanBridgeCfg()
    bridge_ppo_cfg = SiriusTwoSpanBridgeCfgPPO()
    print("   ✓ SiriusTwoSpanBridgeCfg 实例化成功")
    print(f"     - Actor dims: {bridge_ppo_cfg.policy.actor_hidden_dims}")
    print(f"     - Critic dims: {bridge_ppo_cfg.policy.critic_hidden_dims}")
    print(f"     - Activation: {bridge_ppo_cfg.policy.activation}")
    print(f"     - Experiment: {bridge_ppo_cfg.runner.experiment_name}")
except Exception as e:
    print(f"   ✗ 导入或实例化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Verify compatibility
print("\n4. 验证兼容性...")

# Check policy structure
if (flat_ppo_cfg.policy.actor_hidden_dims == bridge_ppo_cfg.policy.actor_hidden_dims and
    flat_ppo_cfg.policy.critic_hidden_dims == bridge_ppo_cfg.policy.critic_hidden_dims and
    flat_ppo_cfg.policy.activation == bridge_ppo_cfg.policy.activation):
    print("   ✓ Policy 配置完全一致")
else:
    print("   ✗ Policy 配置不一致!")
    sys.exit(1)

# Check observation dimensions
if flat_cfg.env.num_observations == bridge_cfg.env.num_observations:
    print(f"   ✓ 观测维度一致: {flat_cfg.env.num_observations}")
else:
    print(f"   ✗ 观测维度不一致: flat={flat_cfg.env.num_observations}, bridge={bridge_cfg.env.num_observations}")
    sys.exit(1)

# Check action dimensions
if flat_cfg.env.num_actions == bridge_cfg.env.num_actions:
    print(f"   ✓ 动作维度一致: {flat_cfg.env.num_actions}")
else:
    print(f"   ✗ 动作维度不一致: flat={flat_cfg.env.num_actions}, bridge={bridge_cfg.env.num_actions}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ 所有测试通过！配置文件正确，可以开始训练。")
print("="*60)
print("\n推荐训练流程:")
print("  1. 阶段1 (平地):")
print("     cd legged_gym/legged_gym")
print("     python scripts/train.py --task=sirius --num_envs=2048 --headless")
print()
print("  2. 阶段2 (桥梁，从阶段1继续):")
print("     python scripts/train.py --task=sirius_diff_vis --resume --headless")
print()
