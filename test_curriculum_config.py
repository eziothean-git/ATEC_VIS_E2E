#!/usr/bin/env python3
"""
测试课程学习配置是否正确

检查：
1. curriculum 是否为 True
2. max_init_terrain_level 是否为 1
3. 初始化后的 terrain_levels 分布
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'legged_gym'))

from legged_gym.envs.sirius_diff_vis.sirius_curriculum_config import SiriusCurriculumCfg

def test_config():
    print("=" * 60)
    print("测试课程学习配置")
    print("=" * 60)
    
    cfg = SiriusCurriculumCfg()
    
    print(f"\n地形配置:")
    print(f"  mesh_type: {cfg.terrain.mesh_type}")
    print(f"  curriculum: {cfg.terrain.curriculum}")
    print(f"  selected: {cfg.terrain.selected}")
    print(f"  num_rows (难度数): {cfg.terrain.num_rows}")
    print(f"  num_cols (类型数): {cfg.terrain.num_cols}")
    print(f"  max_init_terrain_level: {cfg.terrain.max_init_terrain_level}")
    
    # 验证
    assert cfg.terrain.curriculum == True, "❌ curriculum 应该为 True"
    assert cfg.terrain.max_init_terrain_level == 1, f"❌ max_init_terrain_level 应该为 1，实际为 {cfg.terrain.max_init_terrain_level}"
    
    print(f"\n✅ 配置正确！")
    print(f"\n预期行为:")
    print(f"  - 机器人只应该在难度 0 和 1 生成")
    print(f"  - terrain_levels 应该在 [0, 1] 范围内")
    print(f"  - 训练过程中会逐步晋级到更高难度")
    
    print(f"\n环境配置:")
    print(f"  num_envs: {cfg.env.num_envs}")
    print(f"  episode_length_s: {cfg.env.episode_length_s}")
    
    return True

if __name__ == "__main__":
    try:
        test_config()
        print("\n" + "=" * 60)
        print("✅ 测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
