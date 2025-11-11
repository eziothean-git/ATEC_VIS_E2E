#!/usr/bin/env python3
"""
快速测试配置转换是否正常工作
"""

import sys
import os

# Add paths
LEGGED_GYM_ROOT = os.path.join(os.path.dirname(__file__), '../..')
sys.path.append(LEGGED_GYM_ROOT)

from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg

print("=" * 80)
print("测试配置转换")
print("=" * 80)

# 创建配置
cfg = SiriusSharedPPOCfg()

print("\n[1] 原始配置类:")
print(f"  vision_encoder type: {type(cfg.vision_encoder)}")
print(f"  vision_encoder.latent_dim: {cfg.vision_encoder.latent_dim}")
print(f"  policy.use_vision: {cfg.policy.use_vision}")

# 转换为字典
cfg_dict = class_to_dict(cfg)

print("\n[2] 转换后的字典:")
print(f"  'vision_encoder' in cfg_dict: {'vision_encoder' in cfg_dict}")
if 'vision_encoder' in cfg_dict:
    print(f"  vision_encoder type: {type(cfg_dict['vision_encoder'])}")
    print(f"  vision_encoder keys: {list(cfg_dict['vision_encoder'].keys())[:5]}...")  # 前5个key
    print(f"  vision_encoder['latent_dim']: {cfg_dict['vision_encoder'].get('latent_dim')}")

# 测试转换回对象
class VisionEncoderCfg:
    def __init__(self, cfg_dict):
        for key, value in cfg_dict.items():
            setattr(self, key, value)

if 'vision_encoder' in cfg_dict:
    vision_cfg = VisionEncoderCfg(cfg_dict['vision_encoder'])
    print("\n[3] 转换回对象:")
    print(f"  vision_cfg.latent_dim: {vision_cfg.latent_dim}")
    print(f"  vision_cfg.input_height: {vision_cfg.input_height}")
    print(f"  vision_cfg.cnn_layers: {vision_cfg.cnn_layers[:2]}...")  # 前2层

print("\n✓ 配置转换测试完成!")
