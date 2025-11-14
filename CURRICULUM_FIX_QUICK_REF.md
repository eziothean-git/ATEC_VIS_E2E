# 课程学习修复 - 快速参考

## 🎯 核心修改

### 1️⃣ 地形课程：晋级更慢
```python
# 原来：走 4m (50%) 就晋级
# 现在：走 5.6m (70%) + tracking reward > 40% 才晋级
move_up = (distance > terrain_length * 0.7) & (tracking_quality > 0.4)
```

### 2️⃣ 命令课程：阈值降低
```python
# 原来：tracking reward 需要 80% 才扩展速度范围
# 现在：只需 50% 就能扩展
curriculum_threshold = 0.5  # 从 0.8 降低到 0.5
curriculum_step = 0.1       # 从 0.15 降低到 0.1
```

### 3️⃣ 命令采样：全方向支持
```python
# 横向速度范围扩大
lin_vel_y = [-0.3, 0.3]  # 从 ±0.1 扩展到 ±0.3

# 运动模式更平衡
# 60% 直行为主（原来 80%）
# 30% 混合模式（新增）
# 10% 转向为主（原来 20%）

# 前后更平衡
# 70% 前进（原来 90%）
# 30% 后退（原来 10%）
```

---

## 📊 预期训练曲线

### 地形难度 (Command/terrain_level)
```
Iteration    Avg Level    说明
─────────────────────────────────────
0-500        0.5-1.5      在简单地形学习基础步态
500-1000     1.5-3.0      逐步晋级到中等难度
1000-2000    3.0-5.0      挑战中高难度地形
2000+        4.0-7.0      在高难度地形泛化
```

### 速度命令范围 (Command/max_lin_vel)
```
Iteration    Range              说明
──────────────────────────────────────────
0-300        [0, 0.3]          初始范围，学习基础跟随
300-600      [0, 0.4-0.5]      开始扩展，练习中速
600-1000     [0, 0.5-0.6]      继续扩展
1000-1500    [0, 0.6-0.7]      接近高速
1500+        [0, 0.7-0.8]      达到最大速度
```

### Tracking Reward
```
应该稳步上升，不要剧烈震荡
目标：> 10.0（良好）, > 15.0（优秀）
```

---

## 🐛 常见问题

### Q: 地形难度还是涨得太快？
```python
# 在 sirius_curriculum_config.py 的 _update_terrain_curriculum 中
move_up_distance = distance > (self.terrain.env_length * 0.8)  # 提高到 80%
tracking_threshold = 0.5  # 提高到 50%
```

### Q: 命令范围还是不扩展？
```python
# 在 sirius_curriculum_config.py 的配置中
curriculum_threshold = 0.4  # 降低到 40%
```

### Q: 机器人还是只往一个方向走？
检查：
1. TensorBoard 中命令的分布（可添加直方图日志）
2. 确认 `lin_vel_y` 范围已更新到 `[-0.3, 0.3]`
3. 增加横向速度权重：`lin_vel_y = [-0.4, 0.4]`

---

## ✅ 验证清单

运行训练前检查：
- [ ] `sirius_curriculum_config.py` 已更新配置
- [ ] `sirius_curriculum_config.py` 已添加两个新方法
- [ ] `sirius_joystick.py` 的 `_resample_commands()` 已更新
- [ ] 已导入 `numpy as np`

运行训练中监控：
- [ ] TensorBoard 显示 terrain_level 缓慢上升
- [ ] TensorBoard 显示 max_lin_vel 逐步扩展
- [ ] 终端每 1000 步打印课程进度
- [ ] mean_reward 稳步提升

---

## 🚀 开始训练

```bash
cd /home/eziothean/ATEC_VIS_E2E/legged_gym
python scripts/train.py --task=sirius_curriculum --num_envs=512 --headless

# 另一个终端监控
tensorboard --logdir=logs/sirius_curriculum
```

---

## 📝 修改的文件

1. ✅ `sirius_curriculum_config.py`
   - 添加 `import numpy as np`
   - 修改 `commands` 配置
   - 新增 `_update_terrain_curriculum()`
   - 新增 `update_command_curriculum()`

2. ✅ `sirius_joystick.py`
   - 修改 `_resample_commands()`

3. ✅ `CURRICULUM_FIX_SUMMARY.md` (本文档)
   - 详细说明和分析

---

**最后更新**：2025年11月14日
