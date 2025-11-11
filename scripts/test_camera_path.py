#!/usr/bin/env python3
"""
测试相机是否使用 GPU tensor 路径
"""

import sys
import os

# Add legged_gym to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../legged_gym'))

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

def test_camera_path():
    """测试相机路径配置"""
    
    print("=" * 80)
    print("Camera Path Test")
    print("=" * 80)
    
    # Get args
    args = get_args()
    args.task = 'sirius'
    args.num_envs = 16  # 小数量测试
    args.headless = True
    
    # Create environment
    print("\n1. Creating environment...")
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    
    # Check camera config
    print("\n2. Checking camera configuration:")
    if hasattr(env_cfg, 'camera'):
        print(f"   camera.enable = {env_cfg.camera.enable}")
        print(f"   camera.enable_tensors = {getattr(env_cfg.camera, 'enable_tensors', 'NOT SET')}")
    else:
        print("   ❌ No camera config found!")
        return
    
    # Reset environment to initialize camera
    print("\n3. Resetting environment (initializes camera)...")
    env.reset()
    
    # Try to get depth images
    print("\n4. Getting depth images...")
    try:
        depth = env.get_camera_depth_images(as_torch=True)
        print(f"   ✅ Got depth images!")
        print(f"   Shape: {depth.shape}")
        print(f"   Device: {depth.device}")
        print(f"   Dtype: {depth.dtype}")
        print(f"   Min: {depth.min().item():.3f}, Max: {depth.max().item():.3f}")
    except Exception as e:
        print(f"   ❌ Failed to get depth images: {e}")
        return
    
    # Run a few steps to see performance
    print("\n5. Running 10 steps with camera rendering...")
    import time
    start = time.time()
    for i in range(10):
        actions = env.action_space.sample()
        env.step(actions)
    elapsed = time.time() - start
    fps = 10 / elapsed
    print(f"   FPS: {fps:.1f}")
    
    print("\n" + "=" * 80)
    print("✅ Test complete! Check the debug output above for '[Camera Debug]' messages")
    print("   - If you see 'GPU tensor path (FAST)', optimization is working!")
    print("   - If you see 'CPU/NumPy path (SLOW)', need to debug config")
    print("=" * 80)

if __name__ == '__main__':
    test_camera_path()
