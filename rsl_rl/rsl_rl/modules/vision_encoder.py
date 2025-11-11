# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
视觉编码器模块 - 将深度图编码为低维特征向量

用于端到端视觉强化学习，将原始深度图像映射到紧凑的特征表示。
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class SimpleCNNEncoder(nn.Module):
    """
    简单的3层CNN编码器，用于深度图特征提取。
    
    设计用于低分辨率输入 (58x87)，通过卷积和池化逐步降维，
    最终输出固定维度的特征向量。
    
    Args:
        input_height: 输入深度图高度
        input_width: 输入深度图宽度  
        input_channels: 输入通道数（深度图为1）
        latent_dim: 输出特征维度
        cnn_layers: CNN层配置列表
        activation: 激活函数类型
        use_batch_norm: 是否使用batch normalization
        dropout: dropout比率（0表示不使用）
    
    Example:
        >>> encoder = SimpleCNNEncoder(
        ...     input_height=58, input_width=87, 
        ...     latent_dim=32, cnn_layers=[...]
        ... )
        >>> depth = torch.rand(16, 1, 58, 87)  # batch_size=16
        >>> features = encoder(depth)  # (16, 32)
    """
    
    def __init__(
        self,
        input_height: int = 58,
        input_width: int = 87,
        input_channels: int = 1,
        latent_dim: int = 32,
        cnn_layers: list = None,
        activation: str = 'relu',
        use_batch_norm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        
        # 默认3层CNN配置
        if cnn_layers is None:
            cnn_layers = [
                {'out_channels': 16, 'kernel_size': 5, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
                {'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'use_maxpool': True, 'pool_size': 2},
                {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'use_maxpool': False},
            ]
        
        # 选择激活函数
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'elu':
            act_fn = nn.ELU
        elif activation == 'leaky_relu':
            act_fn = lambda: nn.LeakyReLU(0.2)
        else:
            act_fn = nn.ReLU
        
        # 构建CNN层
        layers = []
        in_channels = input_channels
        h, w = input_height, input_width
        
        for i, layer_cfg in enumerate(cnn_layers):
            out_channels = layer_cfg['out_channels']
            kernel_size = layer_cfg['kernel_size']
            stride = layer_cfg.get('stride', 1)
            padding = kernel_size // 2  # 'same' padding效果
            
            # 卷积层
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            ))
            
            # Batch Normalization (可选)
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # 激活函数
            layers.append(act_fn())
            
            # 计算卷积后的尺寸
            h = (h + 2 * padding - kernel_size) // stride + 1
            w = (w + 2 * padding - kernel_size) // stride + 1
            
            # MaxPooling (可选)
            if layer_cfg.get('use_maxpool', False):
                pool_size = layer_cfg.get('pool_size', 2)
                layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
                h = h // pool_size
                w = w // pool_size
            
            # Dropout (可选)
            if dropout > 0 and i < len(cnn_layers) - 1:  # 最后一层不用dropout
                layers.append(nn.Dropout2d(p=dropout))
            
            in_channels = out_channels
            
            # 调试信息：打印每层后的特征图尺寸
            # print(f"Layer {i+1}: out_channels={out_channels}, output_size=({h}, {w})")
        
        self.conv_layers = nn.Sequential(*layers)
        
        # 计算flatten后的维度
        self.flatten_dim = in_channels * h * w
        
        # 全连接层：flatten_dim -> latent_dim
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, latent_dim),
            act_fn()
        )
        
        # 存储最终特征图尺寸（用于调试）
        self.final_feat_h = h
        self.final_feat_w = w
        
    def forward(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            depth_image: (B, 1, H, W) 归一化到[0,1]的深度图
        
        Returns:
            features: (B, latent_dim) 特征向量
        """
        # CNN特征提取
        x = self.conv_layers(depth_image)  # (B, C, H', W')
        
        # Flatten + FC
        features = self.fc(x)  # (B, latent_dim)
        
        return features
    
    def get_output_info(self) -> Dict[str, Any]:
        """返回输出信息（用于调试和验证）"""
        return {
            'flatten_dim': self.flatten_dim,
            'latent_dim': self.latent_dim,
            'final_feat_size': (self.final_feat_h, self.final_feat_w)
        }


def build_vision_encoder(cfg) -> SimpleCNNEncoder:
    """
    从配置构建视觉编码器
    
    Args:
        cfg: 配置对象，包含 vision_encoder 属性
    
    Returns:
        SimpleCNNEncoder 实例
    
    Example:
        >>> from legged_gym.envs.sirius_diff_vis.sirius_shared_model import SiriusSharedPPOCfg
        >>> cfg = SiriusSharedPPOCfg()
        >>> encoder = build_vision_encoder(cfg)
    """
    ve_cfg = cfg.vision_encoder
    
    encoder = SimpleCNNEncoder(
        input_height=ve_cfg.input_height,
        input_width=ve_cfg.input_width,
        input_channels=ve_cfg.input_channels,
        latent_dim=ve_cfg.latent_dim,
        cnn_layers=ve_cfg.cnn_layers,
        activation=ve_cfg.activation,
        use_batch_norm=ve_cfg.use_batch_norm,
        dropout=ve_cfg.dropout
    )
    
    return encoder


# 测试代码
if __name__ == "__main__":
    print("Testing SimpleCNNEncoder...")
    
    # 创建编码器
    encoder = SimpleCNNEncoder(
        input_height=58,
        input_width=87,
        latent_dim=32
    )
    
    # 打印网络结构
    print(f"\nEncoder architecture:")
    print(encoder)
    
    # 打印输出信息
    info = encoder.get_output_info()
    print(f"\nOutput info:")
    print(f"  Flatten dim: {info['flatten_dim']}")
    print(f"  Latent dim: {info['latent_dim']}")
    print(f"  Final feature map size: {info['final_feat_size']}")
    
    # 测试前向传播
    batch_size = 16
    depth = torch.rand(batch_size, 1, 58, 87)
    print(f"\nInput shape: {depth.shape}")
    
    features = encoder(depth)
    print(f"Output shape: {features.shape}")
    
    # 计算参数量
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    print("\n✓ SimpleCNNEncoder test passed!")
