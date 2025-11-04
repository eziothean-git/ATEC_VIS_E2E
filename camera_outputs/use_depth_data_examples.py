#!/usr/bin/env python3
"""
示例：如何使用深度相机数据

演示如何加载和使用 camera_demo.py 生成的各种数据文件
"""

import numpy as np
import matplotlib.pyplot as plt

# 数据文件路径
import os
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def example_1_load_numpy_array():
    """示例 1: 加载 numpy 数组"""
    print("="*60)
    print("示例 1: 加载 numpy 数组文件")
    print("="*60)
    
    # 加载原始深度数据 (Isaac Gym 原始输出，负数)
    depth_raw = np.load(f"{DATA_DIR}/depth_raw_step_0000.npy")
    print(f"原始深度数据形状: {depth_raw.shape}")
    print(f"  - {depth_raw.shape[0]} 个环境")
    print(f"  - 图像大小: {depth_raw.shape[1]}x{depth_raw.shape[2]}")
    print(f"数据类型: {depth_raw.dtype}")
    print(f"原始值范围: [{np.nanmin(depth_raw):.2f}, {np.nanmax(depth_raw):.2f}]")
    
    # 加载绝对值深度数据 (已处理，正数)
    depth_abs = np.load(f"{DATA_DIR}/depth_abs_step_0000.npy")
    print(f"\n绝对值深度数据形状: {depth_abs.shape}")
    valid_depth = depth_abs[~np.isnan(depth_abs)]
    print(f"有效深度范围: [{np.min(valid_depth):.2f}, {np.max(valid_depth):.2f}] m")
    print(f"平均深度: {np.mean(valid_depth):.2f} m")
    
    # 访问特定环境的数据
    env0_depth = depth_abs[0]  # 第一个环境
    print(f"\n环境0的深度图: {env0_depth.shape}")
    
    # 访问特定像素
    center_h, center_w = env0_depth.shape[0] // 2, env0_depth.shape[1] // 2
    center_depth = env0_depth[center_h, center_w]
    print(f"中心像素 [{center_h}, {center_w}] 的深度: {center_depth:.3f} m")
    
    return depth_abs


def example_2_load_csv():
    """示例 2: 加载 CSV 文件 (可用 Excel 打开)"""
    print("\n" + "="*60)
    print("示例 2: 加载 CSV 文件")
    print("="*60)
    
    # 加载 CSV
    depth_matrix = np.loadtxt(f"{DATA_DIR}/depth_matrix_env0_final.csv", 
                              delimiter=',', comments='#')
    print(f"CSV 矩阵形状: {depth_matrix.shape}")
    print(f"这个文件可以直接在 Excel 中打开查看")
    
    # 加载采样的 CSV（更小，更容易查看）
    sampled = np.loadtxt(f"{DATA_DIR}/depth_matrix_env0_sampled.csv",
                        delimiter=',', comments='#')
    print(f"\n采样矩阵形状: {sampled.shape} (每10个像素采样一次)")
    print(f"文件大小更小，更适合在 Excel 中查看")
    
    return depth_matrix


def example_3_analyze_depth():
    """示例 3: 分析深度数据"""
    print("\n" + "="*60)
    print("示例 3: 深度数据分析")
    print("="*60)
    
    depth = np.load(f"{DATA_DIR}/depth_abs_step_0000.npy")[0]  # 第一个环境
    
    # 过滤有效数据
    valid = depth[~np.isnan(depth)]
    
    # 统计分析
    print(f"总像素数: {depth.size}")
    print(f"有效像素数: {len(valid)} ({100*len(valid)/depth.size:.1f}%)")
    print(f"\n深度统计:")
    print(f"  最小值: {np.min(valid):.4f} m")
    print(f"  最大值: {np.max(valid):.4f} m")
    print(f"  平均值: {np.mean(valid):.4f} m")
    print(f"  中位数: {np.median(valid):.4f} m")
    print(f"  标准差: {np.std(valid):.4f} m")
    
    # 距离分布
    print(f"\n距离分布:")
    ranges = [(0, 0.2, "机器人自身"), 
              (0.2, 0.5, "地面(近)"),
              (0.5, 1.0, "近处"),
              (1.0, 2.0, "中距离"),
              (2.0, 5.0, "远处"),
              (5.0, np.inf, "很远")]
    
    for min_d, max_d, label in ranges:
        count = np.sum((valid >= min_d) & (valid < max_d))
        percentage = 100 * count / len(valid)
        print(f"  {min_d:.1f}-{max_d:.1f}m ({label:12s}): {count:5d} 像素 ({percentage:5.1f}%)")


def example_4_visualize():
    """示例 4: 可视化深度图"""
    print("\n" + "="*60)
    print("示例 4: 可视化深度数据")
    print("="*60)
    
    depth = np.load(f"{DATA_DIR}/depth_abs_step_0000.npy")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(min(4, depth.shape[0])):
        ax = axes[i]
        im = ax.imshow(depth[i], cmap='turbo', vmin=0, vmax=3)
        ax.set_title(f'Environment {i}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Depth (m)')
    
    plt.suptitle('Depth Camera Visualization', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{DATA_DIR}/python_visualization.png', dpi=150)
    print("已保存可视化图像到: python_visualization.png")
    plt.close()


def example_5_extract_roi():
    """示例 5: 提取感兴趣区域 (ROI)"""
    print("\n" + "="*60)
    print("示例 5: 提取感兴趣区域")
    print("="*60)
    
    depth = np.load(f"{DATA_DIR}/depth_abs_step_0000.npy")[0]
    h, w = depth.shape
    
    # 提取中心区域 (用于前方障碍物检测)
    center_h, center_w = h // 2, w // 2
    roi_size = 50
    roi_center = depth[center_h-roi_size:center_h+roi_size, 
                      center_w-roi_size:center_w+roi_size]
    
    print(f"原始图像大小: {depth.shape}")
    print(f"中心 ROI 大小: {roi_center.shape}")
    
    valid_roi = roi_center[~np.isnan(roi_center)]
    if len(valid_roi) > 0:
        print(f"中心区域深度范围: [{np.min(valid_roi):.3f}, {np.max(valid_roi):.3f}] m")
        print(f"中心区域平均深度: {np.mean(valid_roi):.3f} m")
        
        # 检测前方是否有障碍物
        threshold = 1.0  # 1米以内认为有障碍物
        close_pixels = np.sum(valid_roi < threshold)
        print(f"\n前方 1m 内有障碍物的像素: {close_pixels}/{len(valid_roi)} "
              f"({100*close_pixels/len(valid_roi):.1f}%)")
    
    return roi_center


def example_6_temporal_analysis():
    """示例 6: 时序分析 (比较不同时刻的深度)"""
    print("\n" + "="*60)
    print("示例 6: 时序分析")
    print("="*60)
    
    # 加载多个时间步的数据
    steps = [0, 30, 60, 90]
    mean_depths = []
    
    for step in steps:
        depth = np.load(f"{DATA_DIR}/depth_abs_step_{step:04d}.npy")[0]
        valid = depth[~np.isnan(depth)]
        mean_depth = np.mean(valid) if len(valid) > 0 else 0
        mean_depths.append(mean_depth)
        print(f"Step {step:3d}: 平均深度 = {mean_depth:.3f} m")
    
    # 绘制时序曲线
    plt.figure(figsize=(10, 5))
    plt.plot(steps, mean_depths, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Step')
    plt.ylabel('Mean Depth (m)')
    plt.title('Average Depth Over Time')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{DATA_DIR}/temporal_analysis.png', dpi=150)
    print(f"\n已保存时序分析图到: temporal_analysis.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("深度相机数据使用示例")
    print("="*60)
    
    # 运行所有示例
    depth_abs = example_1_load_numpy_array()
    depth_matrix = example_2_load_csv()
    example_3_analyze_depth()
    example_4_visualize()
    roi = example_5_extract_roi()
    example_6_temporal_analysis()
    
    print("\n" + "="*60)
    print("所有示例运行完毕！")
    print("="*60)
    print("\n💡 提示:")
    print("  1. 原始数据在 .npy 文件中，使用 np.load() 加载")
    print("  2. CSV 文件可以用 Excel/LibreOffice 直接打开")
    print("  3. 统计信息在 .txt 文件中，可以用文本编辑器查看")
    print("  4. 使用 depth[~np.isnan(depth)] 过滤无效值")
    print("  5. 深度单位是米 (m)")
    print()
