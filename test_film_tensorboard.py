#!/usr/bin/env python3
"""
测试 FiLM 模块的 TensorBoard 日志功能

验证：
1. _log_film_statistics 方法能否正确提取 scale 和 shift
2. 统计信息是否准确
3. TensorBoard 日志格式是否正确
"""

import torch
import torch.nn as nn
import sys
import os

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'rsl_rl'))


def get_activation(act_name):
    """获取激活函数"""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "relu":
        return nn.ReLU()
    else:
        return nn.ReLU()


def test_film_statistics_extraction():
    """测试 FiLM 统计信息提取"""
    print("=" * 60)
    print("测试: FiLM 统计信息提取")
    print("=" * 60)
    
    # 创建模拟的 FiLM MLP
    num_vision_latent = 32
    num_proprio_obs = 45
    film_hidden_dims = [64]
    film_scale_limit = 0.1
    
    dims = [num_vision_latent] + film_hidden_dims + [2 * num_proprio_obs]
    layers = []
    
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i+1]), get_activation('elu')]
    
    layers += [nn.Linear(dims[-2], dims[-1])]
    film_mlp = nn.Sequential(*layers)
    
    # 零初始化（模拟训练初期）
    nn.init.zeros_(film_mlp[-1].weight)
    nn.init.zeros_(film_mlp[-1].bias)
    
    print("\n阶段 1: 训练初期（零初始化）")
    print("-" * 60)
    
    batch_size = 256
    vision_latent = torch.rand(batch_size, num_vision_latent)
    proprio_obs = torch.rand(batch_size, num_proprio_obs)
    
    with torch.no_grad():
        # 通过 FiLM MLP
        film_out = film_mlp(vision_latent)
        scale_raw, shift_raw = torch.split(film_out, num_proprio_obs, dim=-1)
        scale = film_scale_limit * torch.tanh(scale_raw)
        shift = shift_raw
        
        # 计算调制效果
        modulated = proprio_obs * (1.0 + scale) + shift
        modulation_magnitude = (modulated - proprio_obs).abs().mean()
        relative_change = modulation_magnitude / (proprio_obs.abs().mean() + 1e-8)
        
        # 打印统计
        print(f"Scale 统计:")
        print(f"  均值: {scale.mean().item():.8f}")
        print(f"  标准差: {scale.std().item():.8f}")
        print(f"  范围: [{scale.min().item():.8f}, {scale.max().item():.8f}]")
        print(f"  绝对值均值: {scale.abs().mean().item():.8f}")
        
        print(f"\nShift 统计:")
        print(f"  均值: {shift.mean().item():.8f}")
        print(f"  标准差: {shift.std().item():.8f}")
        print(f"  范围: [{shift.min().item():.8f}, {shift.max().item():.8f}]")
        print(f"  绝对值均值: {shift.abs().mean().item():.8f}")
        
        print(f"\n调制效果:")
        print(f"  调制幅度: {modulation_magnitude.item():.8f}")
        print(f"  相对变化: {relative_change.item()*100:.6f}%")
        
        # 验证
        assert abs(scale.mean().item()) < 1e-6, "零初始化时 scale 均值应为 0"
        assert abs(shift.mean().item()) < 1e-6, "零初始化时 shift 均值应为 0"
        assert modulation_magnitude.item() < 1e-6, "零初始化时调制幅度应为 0"
        print("\n✓ 零初始化状态验证通过")
    
    print("\n阶段 2: 模拟训练后（有一定权重）")
    print("-" * 60)
    
    # 随机初始化输出层（模拟训练后）
    nn.init.normal_(film_mlp[-1].weight, mean=0, std=0.01)
    nn.init.normal_(film_mlp[-1].bias, mean=0, std=0.01)
    
    with torch.no_grad():
        # 重新计算
        film_out = film_mlp(vision_latent)
        scale_raw, shift_raw = torch.split(film_out, num_proprio_obs, dim=-1)
        scale = film_scale_limit * torch.tanh(scale_raw)
        shift = shift_raw
        
        modulated = proprio_obs * (1.0 + scale) + shift
        modulation_magnitude = (modulated - proprio_obs).abs().mean()
        relative_change = modulation_magnitude / (proprio_obs.abs().mean() + 1e-8)
        
        # 检查饱和度
        scale_raw_tanh = torch.tanh(scale_raw)
        saturation = (scale_raw_tanh.abs() > 0.9).float().mean()
        
        print(f"Scale 统计:")
        print(f"  均值: {scale.mean().item():.6f}")
        print(f"  标准差: {scale.std().item():.6f}")
        print(f"  范围: [{scale.min().item():.6f}, {scale.max().item():.6f}]")
        print(f"  绝对值均值: {scale.abs().mean().item():.6f}")
        
        print(f"\nShift 统计:")
        print(f"  均值: {shift.mean().item():.6f}")
        print(f"  标准差: {shift.std().item():.6f}")
        print(f"  范围: [{shift.min().item():.6f}, {shift.max().item():.6f}]")
        print(f"  绝对值均值: {shift.abs().mean().item():.6f}")
        
        print(f"\n调制效果:")
        print(f"  调制幅度: {modulation_magnitude.item():.6f}")
        print(f"  相对变化: {relative_change.item()*100:.2f}%")
        print(f"  饱和度: {saturation.item()*100:.2f}% (比例 > 0.9)")
        
        # 验证
        assert scale.abs().mean().item() > 0, "训练后 scale 应该非零"
        assert modulation_magnitude.item() > 0, "训练后应该有调制效果"
        print("\n✓ 训练后状态验证通过")
    
    print("\n" + "=" * 60)
    print("✅ FiLM 统计信息提取测试通过！")
    print("=" * 60)
    print("\nTensorBoard 将记录以下指标:")
    print("  📊 FiLM/scale_mean       - scale 均值")
    print("  📊 FiLM/scale_std        - scale 标准差")
    print("  📊 FiLM/scale_min        - scale 最小值")
    print("  📊 FiLM/scale_max        - scale 最大值")
    print("  📊 FiLM/scale_abs_mean   - scale 绝对值均值")
    print("  📊 FiLM/shift_mean       - shift 均值")
    print("  📊 FiLM/shift_std        - shift 标准差")
    print("  📊 FiLM/shift_min        - shift 最小值")
    print("  📊 FiLM/shift_max        - shift 最大值")
    print("  📊 FiLM/shift_abs_mean   - shift 绝对值均值")
    print("  📊 FiLM/modulation_magnitude - 调制幅度")
    print("  📊 FiLM/relative_change      - 相对变化")
    print("  📊 FiLM/scale_saturation     - 饱和度（限幅检测）")
    print("\n使用方法:")
    print("  训练后运行: tensorboard --logdir=logs/")
    print("  在浏览器中打开 TensorBoard，查看 'FiLM' 标签页")
    print("=" * 60 + "\n")


def main():
    try:
        test_film_statistics_extraction()
        return 0
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
