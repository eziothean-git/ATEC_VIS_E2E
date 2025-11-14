#!/usr/bin/env python3
"""
验证朝向对齐功能是否正确配置和生效

检查项：
1. heading_command 是否启用
2. _resample_commands 是否被正确覆盖
3. 朝向是否与速度方向对齐
"""

import sys
sys.path.append('/home/eziothean/ATEC_VIS_E2E/legged_gym')

from legged_gym.envs.sirius_diff_vis.sirius_curriculum_config import SiriusCurriculumCfg

def main():
    print("="*70)
    print("验证 SiriusCurriculum 朝向对齐配置")
    print("="*70)
    
    # 检查配置
    cfg = SiriusCurriculumCfg()
    
    print("\n[1] 检查 heading_command 配置:")
    print(f"  cfg.commands.heading_command = {cfg.commands.heading_command}")
    if cfg.commands.heading_command:
        print("  ✅ heading_command 已启用")
    else:
        print("  ❌ heading_command 未启用！使用的是角速度模式")
    
    print("\n[2] 检查命令范围:")
    print(f"  lin_vel_x: {cfg.commands.ranges.lin_vel_x}")
    print(f"  lin_vel_y: {cfg.commands.ranges.lin_vel_y}")
    print(f"  ang_vel_yaw: {cfg.commands.ranges.ang_vel_yaw}")
    print(f"  heading: {cfg.commands.ranges.heading}")
    
    print("\n[3] 检查课程配置:")
    print(f"  curriculum: {cfg.commands.curriculum}")
    print(f"  max_curriculum: {cfg.commands.max_curriculum}")
    print(f"  curriculum_threshold: {cfg.commands.curriculum_threshold}")
    
    # 检查方法覆盖
    print("\n[4] 检查方法覆盖:")
    from legged_gym.envs.sirius_diff_vis.sirius_curriculum_config import SiriusCurriculum
    from legged_gym.envs.sirius_diff_vis.sirius_joystick import SiriusJoyFlat
    
    # 获取方法定义的模块
    curriculum_resample = SiriusCurriculum._resample_commands
    base_resample = SiriusJoyFlat._resample_commands
    
    print(f"  SiriusCurriculum._resample_commands 定义在: {curriculum_resample.__module__}")
    print(f"  SiriusJoyFlat._resample_commands 定义在: {base_resample.__module__}")
    
    if curriculum_resample.__module__ != base_resample.__module__:
        print("  ✅ _resample_commands 已被 SiriusCurriculum 覆盖")
    else:
        print("  ❌ _resample_commands 未被覆盖！使用的是基类方法")
    
    print("\n[5] 检查子类继承:")
    from legged_gym.envs.sirius_diff_vis.sirius_curriculum_il_config import SiriusCurriculumIL
    from legged_gym.envs.sirius_diff_vis.sirius_curriculum_finetune_config import SiriusCurriculumFinetune
    
    il_resample = SiriusCurriculumIL._resample_commands
    finetune_resample = SiriusCurriculumFinetune._resample_commands
    
    print(f"  SiriusCurriculumIL._resample_commands 定义在: {il_resample.__module__}")
    print(f"  SiriusCurriculumFinetune._resample_commands 定义在: {finetune_resample.__module__}")
    
    if il_resample.__module__ == curriculum_resample.__module__:
        print("  ✅ SiriusCurriculumIL 继承了正确的 _resample_commands")
    else:
        print("  ⚠️ SiriusCurriculumIL 可能使用了不同的 _resample_commands")
    
    if finetune_resample.__module__ == curriculum_resample.__module__:
        print("  ✅ SiriusCurriculumFinetune 继承了正确的 _resample_commands")
    else:
        print("  ⚠️ SiriusCurriculumFinetune 可能使用了不同的 _resample_commands")
    
    print("\n" + "="*70)
    print("验证完成！")
    print("="*70)
    
    # 总结
    print("\n[总结]")
    all_ok = True
    
    if not cfg.commands.heading_command:
        print("❌ heading_command 未启用")
        all_ok = False
    
    if curriculum_resample.__module__ == base_resample.__module__:
        print("❌ _resample_commands 未被覆盖")
        all_ok = False
    
    if all_ok:
        print("✅ 所有检查通过！朝向对齐功能应该已经生效")
        print("\n建议：启动训练并观察调试输出")
        print("  python legged_gym/scripts/train.py --task=sirius_curriculum --num_envs=64")
        print("\n查看日志中的 [_resample_commands] 和 [Heading Alignment Check] 输出")
    else:
        print("❌ 存在问题，需要修复")
    
    print()

if __name__ == "__main__":
    main()
