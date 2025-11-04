# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Simple demo: enable a depth camera on the robot and fetch frames.

import os
import numpy as np
import matplotlib.pyplot as plt

# Import minimal pieces to avoid circular imports via utils.__init__
from legged_gym.utils.helpers import get_args
from legged_gym.utils.task_registry import task_registry
import legged_gym.envs  # ensures tasks are registered


def run(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # keep it small for camera demo
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 4)
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # enable camera and set an attach body name if you know it
    env_cfg.camera.enable = True
    # Common body names: "trunk" (A1), "base", "base_link" (URDF dependent)
    if getattr(env_cfg.camera, 'body_name', None) is None:
        # try a reasonable default; you can override with --body "name"
        env_cfg.camera.body_name = args.camera_body if hasattr(args, 'camera_body') and args.camera_body else None
    # Camera position: forward 45cm, centered, -3cm high
    env_cfg.camera.position = [0.45, 0.0, -0.03]
    # Pitch down to look at ground (positive pitch = looking down)
    env_cfg.camera.rpy = [0.0, 0.6, 0.0]  # pitch down ~34 degrees
    env_cfg.camera.width = 320
    env_cfg.camera.height = 240

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Set a better viewing angle for the viewer (env.gym already has gymapi loaded)
    env.set_camera(position=[2.5, 2.5, 1.5], lookat=[0, 0, 0.3])
    
    obs = env.get_observations()
    print("\n" + "="*60)
    print("深度相机 Demo - 观察机器狗和相机位置")
    print("="*60)
    print(f"相机配置:")
    print(f"  - 挂载位置: {env_cfg.camera.position} (相对于机器人body)")
    print(f"  - 朝向(roll,pitch,yaw): {env_cfg.camera.rpy}")
    print(f"  - 分辨率: {env_cfg.camera.width}x{env_cfg.camera.height}")
    print(f"  - 视场角: {env_cfg.camera.horizontal_fov}°")
    print("\n" + "="*60)
    print("场景中的可视化标记:")
    print("  🔴 红色球体 = 相机位置")
    print("  🔴 红色箭头 = 相机朝向 (forward)")
    print("  🟢 绿色横线 = 相机左右范围")
    print("  🔵 蓝色竖线 = 相机向上方向")
    print("="*60)
    print("\n提示: 用鼠标在窗口中拖拽旋转视角\n")    # step a few times to settle
    import torch
    import time
    
    out_dir = os.path.join(os.getcwd(), 'camera_outputs')
    os.makedirs(out_dir, exist_ok=True)
    
    # Run for longer and save multiple frames
    num_steps = 300
    save_interval = 30
    visualize_interval = 10  # Update camera markers every 10 steps
    
    for step in range(num_steps):
        # Apply small random actions to make the robot move slightly
        if step < 100:
            # Let it settle first
            actions = torch.zeros((env_cfg.env.num_envs, env_cfg.env.num_actions), dtype=torch.float32, device=env.device)
        else:
            # Small oscillating movements to see the robot
            t = step / 30.0
            actions = torch.sin(torch.tensor([t], device=env.device)).repeat(env_cfg.env.num_envs, env_cfg.env.num_actions) * 0.2
        
        obs, _, _, _, _ = env.step(actions)
        
        # Update camera position visualization periodically
        if step % visualize_interval == 0:
            env.gym.clear_lines(env.viewer)  # Clear old markers
            env.visualize_camera_position()  # Draw new markers at current position
        
        # Save and display depth images at intervals
        if step % save_interval == 0:
            depth = env.get_camera_depth_images(as_torch=False)  # (N, H, W)
            
            # Isaac Gym returns negative depth values - take absolute value
            depth_abs = np.abs(depth)
            depth_abs[np.isinf(depth)] = np.nan  # Replace inf with nan
            
            # Statistics
            valid_depth = depth_abs[~np.isnan(depth_abs)]
            if len(valid_depth) > 0:
                print(f"Step {step:3d}: 深度范围 [{np.min(valid_depth):.2f}, {np.max(valid_depth):.2f}]m | "
                      f"有效像素 {len(valid_depth)}/{depth.size} ({100*len(valid_depth)/depth.size:.1f}%)")
            else:
                print(f"Step {step:3d}: 深度全为 inf (相机没看到任何东西)")
            
            # Save raw depth data as numpy array
            np.save(os.path.join(out_dir, f'depth_raw_step_{step:04d}.npy'), depth)
            np.save(os.path.join(out_dir, f'depth_abs_step_{step:04d}.npy'), depth_abs)
            
            # Save detailed statistics to text file
            stats_file = os.path.join(out_dir, f'depth_stats_step_{step:04d}.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Depth Camera Statistics - Step {step}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Configuration:\n")
                f.write(f"  Resolution: {env_cfg.camera.width} x {env_cfg.camera.height}\n")
                f.write(f"  Camera position: {env_cfg.camera.position}\n")
                f.write(f"  Camera orientation (RPY): {env_cfg.camera.rpy}\n")
                f.write(f"  FOV: {env_cfg.camera.horizontal_fov} degrees\n\n")
                
                f.write(f"Data Shape: {depth.shape}\n")
                f.write(f"  - Number of environments: {depth.shape[0]}\n")
                f.write(f"  - Image height: {depth.shape[1]}\n")
                f.write(f"  - Image width: {depth.shape[2]}\n")
                f.write(f"  - Total pixels per env: {depth.shape[1] * depth.shape[2]}\n\n")
                
                for env_idx in range(min(4, depth.shape[0])):
                    f.write(f"Environment {env_idx}:\n")
                    env_depth = depth_abs[env_idx]
                    valid = env_depth[~np.isnan(env_depth)]
                    
                    if len(valid) > 0:
                        f.write(f"  Valid pixels: {len(valid)} / {env_depth.size} ({100*len(valid)/env_depth.size:.1f}%)\n")
                        f.write(f"  Depth range: [{np.min(valid):.4f}, {np.max(valid):.4f}] m\n")
                        f.write(f"  Mean depth: {np.mean(valid):.4f} m\n")
                        f.write(f"  Median depth: {np.median(valid):.4f} m\n")
                        f.write(f"  Std deviation: {np.std(valid):.4f} m\n")
                        f.write(f"  Percentiles:\n")
                        f.write(f"    10%: {np.percentile(valid, 10):.4f} m\n")
                        f.write(f"    25%: {np.percentile(valid, 25):.4f} m\n")
                        f.write(f"    50%: {np.percentile(valid, 50):.4f} m\n")
                        f.write(f"    75%: {np.percentile(valid, 75):.4f} m\n")
                        f.write(f"    90%: {np.percentile(valid, 90):.4f} m\n")
                        
                        # Depth distribution
                        hist, bins = np.histogram(valid, bins=[0, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf])
                        f.write(f"  Depth distribution:\n")
                        f.write(f"    0.0-0.2m: {hist[0]} pixels ({100*hist[0]/len(valid):.1f}%) - Very close (robot body)\n")
                        f.write(f"    0.2-0.5m: {hist[1]} pixels ({100*hist[1]/len(valid):.1f}%) - Close (ground)\n")
                        f.write(f"    0.5-1.0m: {hist[2]} pixels ({100*hist[2]/len(valid):.1f}%) - Near\n")
                        f.write(f"    1.0-2.0m: {hist[3]} pixels ({100*hist[3]/len(valid):.1f}%) - Medium\n")
                        f.write(f"    2.0-5.0m: {hist[4]} pixels ({100*hist[4]/len(valid):.1f}%) - Far\n")
                        f.write(f"    >5.0m:    {hist[5]} pixels ({100*hist[5]/len(valid):.1f}%) - Very far\n")
                        
                        # Sample values from center region
                        h, w = env_depth.shape
                        center_h, center_w = h // 2, w // 2
                        f.write(f"\n  Center region samples (5x5 around center pixel [{center_h}, {center_w}]):\n")
                        for dh in range(-2, 3):
                            row_vals = []
                            for dw in range(-2, 3):
                                val = env_depth[center_h + dh, center_w + dw]
                                if np.isnan(val):
                                    row_vals.append("  nan ")
                                else:
                                    row_vals.append(f"{val:6.3f}")
                            f.write(f"    {' '.join(row_vals)}\n")
                    else:
                        f.write(f"  No valid depth data (all inf/nan)\n")
                    f.write("\n")
                
                f.write("\nFile paths:\n")
                f.write(f"  Raw depth (original): depth_raw_step_{step:04d}.npy\n")
                f.write(f"  Absolute depth: depth_abs_step_{step:04d}.npy\n")
                f.write(f"  Visualization: depth_step_{step:04d}.png\n")
            
            print(f"  → 已保存: depth_raw/abs_step_{step:04d}.npy + stats.txt")
            
            # Save with better visualization
            fig, axes = plt.subplots(1, min(4, env_cfg.env.num_envs), figsize=(15, 4))
            if env_cfg.env.num_envs == 1:
                axes = [axes]
            
            for i in range(min(4, env_cfg.env.num_envs)):
                # Visualize absolute depth with better range
                im = axes[i].imshow(depth_abs[i], cmap='turbo', vmin=0, vmax=3)
                axes[i].set_title(f'Env {i} - Step {step}', fontsize=10)
                axes[i].axis('off')
                
                # Add text annotation
                axes[i].text(0.5, 0.98, '圆形 = 机器狗自己的腿/身体', 
                           transform=axes[i].transAxes, fontsize=8,
                           verticalalignment='top', horizontalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Add colorbar for each subplot
                cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                cbar.set_label('Distance (m)', rotation=270, labelpad=15, fontsize=8)
            
            plt.suptitle(f'Depth Camera View - Camera @{env_cfg.camera.position} m', fontsize=11, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'depth_step_{step:04d}.png'), dpi=120, bbox_inches='tight')
            plt.close()
        
        time.sleep(0.005)  # Small delay to see the visualization
    
    # Get final depth data and save as CSV for easy viewing
    print("\n正在生成 CSV 文件用于 Excel 查看...")
    final_depth = env.get_camera_depth_images(as_torch=False)
    final_depth_abs = np.abs(final_depth)
    final_depth_abs[np.isinf(final_depth)] = np.nan
    
    # Save first environment's depth as CSV
    csv_file = os.path.join(out_dir, 'depth_matrix_env0_final.csv')
    np.savetxt(csv_file, final_depth_abs[0], delimiter=',', fmt='%.4f', 
               header=f'Depth matrix (m) - {final_depth_abs[0].shape[0]}x{final_depth_abs[0].shape[1]} - Camera @{env_cfg.camera.position}',
               comments='# ')
    
    # Save a smaller sampled version for easier viewing (every 10th pixel)
    sampled = final_depth_abs[0][::10, ::10]
    csv_sampled = os.path.join(out_dir, 'depth_matrix_env0_sampled.csv')
    np.savetxt(csv_sampled, sampled, delimiter=',', fmt='%.4f',
               header=f'Sampled depth matrix (every 10 pixels) - {sampled.shape[0]}x{sampled.shape[1]}',
               comments='# ')
    
    print("\n" + "="*60)
    print(f"✓ Demo 完成！所有数据已保存到: {out_dir}")
    print("="*60)
    print(f"📊 数据文件:")
    print(f"  - 图片: depth_step_*.png ({num_steps//save_interval + 1} 张)")
    print(f"  - 原始矩阵: depth_raw_step_*.npy")
    print(f"  - 绝对值矩阵: depth_abs_step_*.npy")
    print(f"  - 统计信息: depth_stats_step_*.txt")
    print(f"  - CSV矩阵(完整): depth_matrix_env0_final.csv")
    print(f"  - CSV矩阵(采样): depth_matrix_env0_sampled.csv")
    print("\n💡 如何使用:")
    print(f"  - Python: data = np.load('depth_abs_step_0000.npy')")
    print(f"  - Excel: 打开 depth_matrix_env0_*.csv 文件")
    print(f"  - 统计: 查看 depth_stats_step_*.txt 文本文件")
    print("="*60)


if __name__ == '__main__':
    args = get_args()
    # You can pass --task a1 or other registered tasks
    # Optionally add --camera_body trunk/base/base_link by extending get_args in utils if needed
    run(args)
