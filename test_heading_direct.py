#!/usr/bin/env python3
"""直接测试 heading 对齐功能 - 只运行10步"""
import sys
import os

# 设置路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'legged_gym'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rsl_rl'))

print("="*80)
print("🔥 开始测试 SiriusCurriculum heading 对齐")
print("="*80)

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import task_registry

print(f"\n✅ 成功导入 task_registry")
print(f"   可用任务: {list(task_registry.task_classes.keys())}")

# 创建环境
env_cfg, train_cfg = task_registry.get_cfgs("sirius_curriculum")
print(f"\n✅ 成功获取配置")
print(f"   heading_command = {env_cfg.commands.heading_command}")

# 准备仿真参数
from isaacgym import gymapi
sim_params = gymapi.SimParams()
sim_params.dt = env_cfg.sim.dt
sim_params.substeps = env_cfg.sim.substeps
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

print(f"\n🚀 创建环境 (8个环境)...")
env, _ = task_registry.make_env("sirius_curriculum", args=None, env_cfg=env_cfg)
print(f"✅ 环境创建成功")

# 获取环境信息
print(f"\n📊 环境信息:")
print(f"   环境数量: {env.num_envs}")
print(f"   观测维度: {env.num_obs}")
print(f"   动作维度: {env.num_actions}")
print(f"   环境类型: {type(env).__name__}")
print(f"   父类: {type(env).__bases__[0].__name__}")

# 检查方法来源
import inspect
callback_method = getattr(env, '_post_physics_step_callback')
resample_method = getattr(env, '_resample_commands')

print(f"\n🔍 方法检查:")
print(f"   _post_physics_step_callback 来源: {inspect.getfile(callback_method)}")
print(f"   _post_physics_step_callback 行号: {inspect.getsourcelines(callback_method)[1]}")
print(f"   _resample_commands 来源: {inspect.getfile(resample_method)}")
print(f"   _resample_commands 行号: {inspect.getsourcelines(resample_method)[1]}")

# 重置环境
print(f"\n🔄 重置环境...")
obs = env.reset()
print(f"✅ 环境重置完成")

# 运行10步
print(f"\n▶️  运行10步，观察打印输出...")
print("="*80)

import torch
actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)

for step in range(10):
    obs, rewards, dones, infos = env.step(actions)
    print(f"   Step {step} completed")

print("="*80)
print(f"\n✅ 测试完成！")
print(f"   如果上面看到 🔥 打印，说明方法被正确调用")
print(f"   如果没看到，说明方法没有被覆盖或者类没有被正确使用")
print("="*80)
