#!/usr/bin/env python3
"""
验证 sirius 和 sirius_diff_vis 任务的模型兼容性。

检查项：
1. 观测维度是否一致
2. 动作维度是否一致  
3. 网络结构（actor/critic hidden dims）是否一致
4. 激活函数是否一致

用法：
    python scripts/verify_model_compatibility.py
"""

import sys
import os

# Add legged_gym to path
LEGGED_GYM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LEGGED_GYM_ROOT)

from legged_gym.utils.task_registry import task_registry


def verify_compatibility():
    """验证两个任务的兼容性"""
    
    print("=" * 60)
    print("验证 task=sirius 和 task=sirius_diff_vis 的模型兼容性")
    print("=" * 60)
    print()
    
    # 获取两个任务的配置
    print("加载任务配置...")
    env_cfg_1, train_cfg_1 = task_registry.get_cfgs(name="sirius")
    env_cfg_2, train_cfg_2 = task_registry.get_cfgs(name="sirius_diff_vis")
    print("✓ 配置加载成功")
    print()
    
    # 检查标志
    all_passed = True
    
    # 1. 检查观测维度
    print("1. 检查观测维度...")
    obs_dim_1 = env_cfg_1.env.num_observations
    obs_dim_2 = env_cfg_2.env.num_observations
    
    print(f"   - sirius:          {obs_dim_1}")
    print(f"   - sirius_diff_vis: {obs_dim_2}")
    
    if obs_dim_1 == obs_dim_2:
        print("   ✓ 观测维度一致")
    else:
        print(f"   ✗ 观测维度不一致! 无法 resume!")
        all_passed = False
    print()
    
    # 2. 检查动作维度
    print("2. 检查动作维度...")
    act_dim_1 = env_cfg_1.env.num_actions
    act_dim_2 = env_cfg_2.env.num_actions
    
    print(f"   - sirius:          {act_dim_1}")
    print(f"   - sirius_diff_vis: {act_dim_2}")
    
    if act_dim_1 == act_dim_2:
        print("   ✓ 动作维度一致")
    else:
        print(f"   ✗ 动作维度不一致! 无法 resume!")
        all_passed = False
    print()
    
    # 3. 检查网络结构
    print("3. 检查网络结构...")
    
    # Actor
    actor_dims_1 = train_cfg_1.policy.actor_hidden_dims
    actor_dims_2 = train_cfg_2.policy.actor_hidden_dims
    
    print(f"   Actor hidden dims:")
    print(f"   - sirius:          {actor_dims_1}")
    print(f"   - sirius_diff_vis: {actor_dims_2}")
    
    if actor_dims_1 == actor_dims_2:
        print("   ✓ Actor 结构一致")
    else:
        print(f"   ✗ Actor 结构不一致! 无法 resume!")
        all_passed = False
    
    # Critic
    critic_dims_1 = train_cfg_1.policy.critic_hidden_dims
    critic_dims_2 = train_cfg_2.policy.critic_hidden_dims
    
    print(f"   Critic hidden dims:")
    print(f"   - sirius:          {critic_dims_1}")
    print(f"   - sirius_diff_vis: {critic_dims_2}")
    
    if critic_dims_1 == critic_dims_2:
        print("   ✓ Critic 结构一致")
    else:
        print(f"   ✗ Critic 结构不一致! 无法 resume!")
        all_passed = False
    print()
    
    # 4. 检查激活函数
    print("4. 检查激活函数...")
    activation_1 = train_cfg_1.policy.activation
    activation_2 = train_cfg_2.policy.activation
    
    print(f"   - sirius:          {activation_1}")
    print(f"   - sirius_diff_vis: {activation_2}")
    
    if activation_1 == activation_2:
        print("   ✓ 激活函数一致")
    else:
        print(f"   ✗ 激活函数不一致! 可能影响 resume!")
        # 这个不是致命问题，只是警告
    print()
    
    # 5. 检查实验名称（信息性）
    print("5. 实验名称（仅供参考）...")
    exp_name_1 = train_cfg_1.runner.experiment_name
    exp_name_2 = train_cfg_2.runner.experiment_name
    
    print(f"   - sirius:          {exp_name_1}")
    print(f"   - sirius_diff_vis: {exp_name_2}")
    print("   (实验名称不同是正常的，用于区分任务)")
    print()
    
    # 总结
    print("=" * 60)
    if all_passed:
        print("✓ 所有检查通过！两个任务可以使用 --resume 无缝衔接。")
        print()
        print("建议的训练流程:")
        print("  1. 阶段1 (平地):")
        print("     python scripts/train.py --task=sirius --headless")
        print()
        print("  2. 阶段2 (桥梁，从阶段1继续):")
        print("     python scripts/train.py --task=sirius_diff_vis --resume \\")
        print(f"         --load_run=-1 --checkpoint=-1 --headless")
        print()
        print("  注意: resume 时会自动从 experiment_name='sirius_flat' 加载权重")
    else:
        print("✗ 存在不兼容问题! 请修复后再尝试 resume。")
        return False
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = verify_compatibility()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ 验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
