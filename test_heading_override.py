#!/usr/bin/env python3
"""
快速验证 SiriusCurriculum 是否正确覆盖了父类方法
"""
import inspect

# 检查方法定义位置
print("="*70)
print("🔍 检查方法覆盖情况")
print("="*70)

# 导入类
import sys
sys.path.insert(0, 'legged_gym')

try:
    # 注意：这里会因为 isaacgym 导入失败，但我们可以直接读取源码
    from legged_gym.envs.sirius_diff_vis.sirius_curriculum_config import SiriusCurriculum
    print("✅ 成功导入 SiriusCurriculum")
except ImportError as e:
    print(f"⚠️  导入失败（预期，因为 isaacgym）: {e}")
    print("\n改用源码检查...")
    
    # 直接检查源码
    import re
    
    curriculum_file = 'legged_gym/legged_gym/envs/sirius_diff_vis/sirius_curriculum_config.py'
    joystick_file = 'legged_gym/legged_gym/envs/sirius_diff_vis/sirius_joystick.py'
    
    print(f"\n📄 检查文件: {curriculum_file}")
    print("-"*70)
    
    # 读取文件
    with open(curriculum_file, 'r', encoding='utf-8') as f:
        curriculum_content = f.read()
    
    with open(joystick_file, 'r', encoding='utf-8') as f:
        joystick_content = f.read()
    
    # 查找类定义和方法定义
    def find_class_methods(content, class_name):
        """查找类中定义的方法"""
        # 找到类定义的开始
        class_pattern = rf'^class {class_name}\([^)]+\):'
        class_match = re.search(class_pattern, content, re.MULTILINE)
        
        if not class_match:
            return []
        
        class_start = class_match.start()
        
        # 找到下一个类定义（或文件结束）
        next_class_pattern = r'^class \w+'
        next_class_matches = list(re.finditer(next_class_pattern, content[class_start + len(class_match.group()):], re.MULTILINE))
        
        if next_class_matches:
            class_end = class_start + len(class_match.group()) + next_class_matches[0].start()
        else:
            class_end = len(content)
        
        # 在类定义范围内查找方法
        class_content = content[class_start:class_end]
        method_pattern = r'^\s{4}def (\w+)\('
        methods = []
        
        for match in re.finditer(method_pattern, class_content, re.MULTILINE):
            method_name = match.group(1)
            # 计算实际行号
            line_num = content[:class_start + match.start()].count('\n') + 1
            methods.append((method_name, line_num))
        
        return methods
    
    # 查找 SiriusCurriculum 的方法
    curriculum_methods = find_class_methods(curriculum_content, 'SiriusCurriculum')
    print(f"\n✅ SiriusCurriculum 类定义的方法：")
    for method_name, line_num in curriculum_methods:
        print(f"   - {method_name:40s} (line {line_num})")
    
    # 查找 SiriusJoyFlat 的方法
    joystick_methods = find_class_methods(joystick_content, 'SiriusJoyFlat')
    print(f"\n📋 SiriusJoyFlat 类定义的方法（部分）：")
    relevant_methods = [m for m in joystick_methods if m[0] in [
        '_post_physics_step_callback', '_resample_commands', 'post_physics_step'
    ]]
    for method_name, line_num in relevant_methods:
        print(f"   - {method_name:40s} (line {line_num})")
    
    # 检查覆盖情况
    print(f"\n{'='*70}")
    print("🎯 方法覆盖检查")
    print("="*70)
    
    curriculum_method_names = {m[0] for m in curriculum_methods}
    joystick_method_names = {m[0] for m in joystick_methods}
    
    critical_methods = ['_post_physics_step_callback', '_resample_commands']
    
    for method in critical_methods:
        in_curriculum = method in curriculum_method_names
        in_joystick = method in joystick_method_names
        
        if in_curriculum and in_joystick:
            print(f"✅ {method:40s} - 已覆盖")
        elif in_joystick and not in_curriculum:
            print(f"❌ {method:40s} - 未覆盖（将使用父类实现）")
        elif in_curriculum and not in_joystick:
            print(f"🆕 {method:40s} - 新增方法")
        else:
            print(f"⚠️  {method:40s} - 两个类都没有定义")
    
    # 检查 heading_command 配置
    print(f"\n{'='*70}")
    print("⚙️  配置检查")
    print("="*70)
    
    heading_matches = re.findall(r'heading_command\s*=\s*(\w+)', curriculum_content)
    if heading_matches:
        print(f"✅ heading_command 配置: {heading_matches[0]}")
    else:
        print(f"❌ 未找到 heading_command 配置")
    
    # 检查是否有调试输出
    debug_patterns = [
        (r'\[_post_physics_step_callback\]', '_post_physics_step_callback 调试输出'),
        (r'\[_resample_commands\]', '_resample_commands 调试输出'),
        (r'\[Heading Alignment Check\]', '朝向对齐检查输出'),
        (r'\[Heading to AngVel Conversion\]', '朝向到角速度转换输出'),
    ]
    
    print(f"\n📊 调试输出检查")
    print("-"*70)
    for pattern, desc in debug_patterns:
        if re.search(pattern, curriculum_content):
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc} - 未找到")
    
    print(f"\n{'='*70}")
    print("📝 结论")
    print("="*70)
    
    if '_post_physics_step_callback' in curriculum_method_names and '_resample_commands' in curriculum_method_names:
        print("✅ 所有关键方法都已正确覆盖")
        print("✅ 现在运行训练应该会看到调试输出")
        print("\n💡 建议运行：")
        print("   python legged_gym/scripts/train.py --task=sirius_curriculum --num_envs=64")
    else:
        print("❌ 方法覆盖不完整，需要修复")
        if '_post_physics_step_callback' not in curriculum_method_names:
            print("   - 缺少 _post_physics_step_callback 覆盖")
        if '_resample_commands' not in curriculum_method_names:
            print("   - 缺少 _resample_commands 覆盖")
    
    print("="*70)
