# 离地惩罚功能添加说明

## 📝 功能描述

添加了新的奖励项 `feet_contact_number`，用于惩罚接地腿数少于2条的情况。

## 🎯 目的

- 鼓励机器人保持稳定的多足支撑
- 防止不稳定的行为（如跳跃、单腿平衡等）
- 提高步态的安全性和可靠性

## ⚙️ 实现细节

### 配置文件修改

**文件**: `sirius_curriculum_config.py`

在 `rewards.scales` 中添加：
```python
# 实现路径: sirius_joystick.py::_reward_feet_contact_number
# 公式: 1{ num_contact_feet < 2 }
# 周期: 每步
# 说明: 惩罚接地腿数少于2条的情况，鼓励稳定的多足支撑
feet_contact_number = -2.0
```

### 奖励函数实现

**文件**: `sirius_joystick.py`

新增方法 `_reward_feet_contact_number()`:
```python
def _reward_feet_contact_number(self):
    """
    Penalize having less than 2 feet in contact with ground.
    
    This encourages stable multi-foot support and discourages risky behaviors
    like jumping or balancing on single foot.
    
    Returns:
        torch.Tensor: Per-env penalty (0 if >= 2 feet in contact, 1 if < 2 feet in contact)
                      Shape: (num_envs,)
    """
    # Determine which feet are in contact (normal force > 1N)
    contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    
    # Count number of feet in contact for each environment
    num_contact_feet = torch.sum(contact, dim=1)
    
    # Return 1.0 (penalty) if less than 2 feet in contact, 0.0 otherwise
    # This will be multiplied by the negative weight in config (e.g., -2.0)
    return (num_contact_feet < 2).float()
```

## 📊 工作原理

1. **接地检测**: 通过检查足部的法向接触力（Z轴），判断哪些脚在地面上
   - 接触条件: `contact_forces[:, feet_indices, 2] > 1.0` (法向力 > 1N)

2. **计数统计**: 统计每个环境中有多少条腿在地面上
   - `num_contact_feet = torch.sum(contact, dim=1)`

3. **惩罚判定**: 如果接地腿数 < 2，返回 1.0（触发惩罚），否则返回 0.0
   - `(num_contact_feet < 2).float()`

4. **权重应用**: 配置权重 `-2.0` 会自动乘以 `dt`（控制频率）
   - 实际每步惩罚 = `-2.0 × dt × 1.0` (当触发时)
   - 假设 `dt = 0.02s`，则每步惩罚约 `-0.04`

## 🔧 参数调整

如果需要调整惩罚强度，可以修改配置中的权重：

```python
# 更严格的惩罚（更不允许离地）
feet_contact_number = -5.0

# 较轻的惩罚（允许偶尔离地）
feet_contact_number = -1.0

# 更宽松的阈值（允许单腿支撑）
# 需要修改代码: (num_contact_feet < 1).float()
```

## 📈 预期效果

训练后，机器人应该：
- ✅ 保持至少2条腿在地面上的时间增加
- ✅ 减少跳跃和不稳定行为
- ✅ 步态更加平稳可靠
- ✅ 提高在崎岖地形上的稳定性

可以在 TensorBoard 中监控：
- `Rewards/feet_contact_number`: 惩罚值（越接近0越好）
- 该值应该随训练逐步减小

## ✅ 验证

添加完成后，代码会自动：
1. 在环境初始化时注册该奖励函数
2. 每个仿真步自动计算
3. 权重会自动乘以 `dt`
4. 结果会累加到总奖励中

无需额外配置，直接训练即可生效！

---

**添加日期**: 2025年11月14日
