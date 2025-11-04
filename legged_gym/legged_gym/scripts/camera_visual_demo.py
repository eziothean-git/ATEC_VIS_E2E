# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Enhanced camera demo with visualization of camera position and live depth view

import os
import numpy as np

# Import Isaac Gym stuff FIRST before any torch/cv2/matplotlib
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry
import legged_gym.envs  # ensures tasks are registered


def run(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # Use just 1 env for clearer visualization
    env_cfg.env.num_envs = 1
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # enable camera and set an attach body name if you know it
    env_cfg.camera.enable = True
    env_cfg.camera.body_name = "base"  # Use base since trunk lookup has issues
    
    # Camera mounted on top-front of the robot
    env_cfg.camera.position = [0.30, 0.0, 0.15]  # Forward, centered, up
    # Point camera forward and slightly down
    env_cfg.camera.rpy = [0.0, 0.2, 0.0]  # Pitch down ~11 degrees to see ground
    env_cfg.camera.width = 640
    env_cfg.camera.height = 480
    env_cfg.camera.horizontal_fov = 87

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Set viewer to see the robot from side angle
    env.set_camera(position=[2.0, 2.0, 1.5], lookat=[0, 0, 0.3])
    
    print("\n" + "="*70)
    print("                深度相机可视化 Demo")
    print("="*70)
    print(f"\n📷 相机配置:")
    print(f"   挂载body: {env_cfg.camera.body_name}")
    print(f"   相对位置: {env_cfg.camera.position} (前, 左, 上)")
    print(f"   朝向(RPY): {[f'{r:.2f}' for r in env_cfg.camera.rpy]} rad")
    print(f"   分辨率:   {env_cfg.camera.width}x{env_cfg.camera.height}")
    print(f"   视场角:   {env_cfg.camera.horizontal_fov}°")
    print(f"\n🎮 控制:")
    print(f"   - Isaac Gym 窗口可以用鼠标旋转视角")
    print(f"   - 按 ESC 或 Q 退出")
    print(f"   - OpenCV 窗口显示实时深度图")
    print("\n" + "="*70 + "\n")

    obs = env.get_observations()
    
    # Now safe to import torch, cv2, matplotlib after env is created
    import torch
    import time
    import cv2
    import matplotlib.pyplot as plt
    
    out_dir = os.path.join(os.getcwd(), 'camera_outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    # Create OpenCV window for live depth view
    cv2.namedWindow('Camera Depth View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Depth View', 960, 720)
    
    step = 0
    try:
        while True:
            # Small random actions to make robot move a bit
            if step < 100:
                actions = torch.zeros((1, env_cfg.env.num_actions), dtype=torch.float32, device=env.device)
            else:
                # Small oscillating motion
                t = step / 50.0
                actions = torch.sin(torch.tensor([t])).repeat(1, env_cfg.env.num_actions) * 0.3
                actions = actions.to(dtype=torch.float32, device=env.device)
            
            obs, _, _, _, _ = env.step(actions)
            
            # Get and display depth image every few steps
            if step % 5 == 0:
                depth = env.get_camera_depth_images(as_torch=False)  # (1, H, W)
                depth_img = depth[0]
                
                # Convert depth to displayable format
                # Replace -inf with max value for visualization
                depth_display = depth_img.copy()
                depth_display[np.isinf(depth_display)] = 10.0  # Far plane
                depth_display = np.clip(depth_display, 0, 10.0)
                
                # Normalize to 0-255 for display
                depth_normalized = ((depth_display / 10.0) * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
                
                # Add info text overlay
                info_text = [
                    f"Step: {step}",
                    f"Depth range: [{np.nanmin(depth_img):.2f}, {np.nanmax(depth_img):.2f}]m",
                    f"Camera pos: {env_cfg.camera.position}",
                    "Press Q to quit"
                ]
                
                y_offset = 30
                for i, text in enumerate(info_text):
                    cv2.putText(depth_colored, text, (10, y_offset + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Camera Depth View', depth_colored)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    print("\n退出...")
                    break
                
                if step % 50 == 0:
                    print(f"Step {step:4d} | Depth: [{np.nanmin(depth_img):6.2f}, {np.nanmax(depth_img):6.2f}]m")
            
            step += 1
            time.sleep(0.01)
            
            if step >= 500:
                break
                
    except KeyboardInterrupt:
        print("\n\n收到中断信号，退出...")
    
    finally:
        cv2.destroyAllWindows()
        
        # Save final frame
        depth = env.get_camera_depth_images(as_torch=False)
        depth_img = depth[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Raw depth
        im1 = ax1.imshow(depth_img, cmap='turbo')
        ax1.set_title(f'原始深度图 (Step {step})')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, label='Depth (m)')
        
        # Clipped for better visualization
        depth_clipped = np.clip(depth_img, 0, 5)
        depth_clipped[np.isinf(depth_img)] = 5
        im2 = ax2.imshow(depth_clipped, cmap='turbo', vmin=0, vmax=5)
        ax2.set_title('深度图 (0-5m)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, label='Depth (m)')
        
        plt.tight_layout()
        final_path = os.path.join(out_dir, 'depth_final_visual.png')
        plt.savefig(final_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n" + "="*70)
        print(f"✓ Demo 完成！最终深度图已保存到: {final_path}")
        print("="*70)


if __name__ == '__main__':
    args = get_args()
    run(args)
