# Curriculum 日志优化

## 问题
之前每次环境重置都会打印 `[CurriculumDebug-Sirius]` 信息，导致终端输出被刷屏：
```
[CurriculumDebug-Sirius] pval=2.917634 mean=2.046265 ema=2.033849 consec=1 resets_since_last_expand=1
[CurriculumDebug-Sirius] pval=2.912819 mean=2.038014 ema=2.034682 consec=2 resets_since_last_expand=2
[CurriculumDebug-Sirius] pval=2.945990 mean=2.071442 ema=2.042034 consec=3 resets_since_last_expand=3
...
```

## 优化方案

### 1. 智能调试日志
现在只在以下情况打印 `[CurriculumDebug]`：
- ✅ **状态变化时**：consec 计数增加或重置为 0
- ✅ **定期汇总**：每 50 次重置打印一次
- ✅ **接近阈值时**：consec 距离扩展阈值不到 2 时

输出格式更简洁：
```
[CurriculumDebug] pval=2.945 mean=2.071 ema=2.042 consec=3/5 resets=3/20
```
- `consec=3/5`：当前连续成功 3 次，需要 5 次
- `resets=3/20`：距上次扩展 3 次重置，最少需要 20 次

### 2. 增强的扩展通知
当成功扩展命令范围时，输出更详细的信息：
```
============================================================
[Curriculum] ✓ EXPANDED COMMAND RANGE
  pval=2.910 (thresh=0.800), ema=2.031
  lin_vel_x: 0.980 → 1.000 m/s
  lin_vel_y: [-1.000, 1.000] m/s
  ang_vel_yaw: [-1.000, 1.000] rad/s
============================================================
```

## 效果对比

### 优化前：
```
[CurriculumDebug-Sirius] pval=2.917634 mean=2.046265 ema=2.033849 consec=1 resets_since_last_expand=1
[CurriculumDebug-Sirius] pval=2.912819 mean=2.038014 ema=2.034682 consec=2 resets_since_last_expand=2
[CurriculumDebug-Sirius] pval=2.945990 mean=2.071442 ema=2.042034 consec=3 resets_since_last_expand=3
[CurriculumDebug-Sirius] pval=2.953117 mean=2.067749 ema=2.047177 consec=4 resets_since_last_expand=4
[CurriculumDebug-Sirius] pval=2.961234 mean=2.074152 ema=2.052398 consec=5 resets_since_last_expand=5
[Curriculum] Expanded commands: pval=2.910 ema=2.031 new_max_x=1.000
```
每次重置都打印 → **刷屏**

### 优化后：
```
[CurriculumDebug] pval=2.917 mean=2.046 ema=2.034 consec=1/5 resets=1/20
[CurriculumDebug] pval=2.912 mean=2.038 ema=2.035 consec=0/5 resets=2/20
... (50 次重置后)
[CurriculumDebug] pval=2.945 mean=2.071 ema=2.042 consec=3/5 resets=50/20
[CurriculumDebug] pval=2.953 mean=2.068 ema=2.047 consec=4/5 resets=51/20  <- 接近阈值
[CurriculumDebug] pval=2.961 mean=2.074 ema=2.052 consec=5/5 resets=52/20

============================================================
[Curriculum] ✓ EXPANDED COMMAND RANGE
  pval=2.910 (thresh=0.800), ema=2.031
  lin_vel_x: 0.980 → 1.000 m/s
  lin_vel_y: [-1.000, 1.000] m/s
  ang_vel_yaw: [-1.000, 1.000] rad/s
============================================================
```
只在关键时刻打印 → **清爽易读**

## 配置参数

如果需要调整打印频率，修改 `sirius_joystick.py` 中的：

```python
# 每 N 次重置打印一次汇总（默认 50）
self.curriculum_debug_counter % 50 == 0

# 距离阈值多近开始打印（默认 2）
self.curriculum_consec_ok_count >= max(1, consec_required - 2)
```

## 完全禁用调试日志

如果不需要任何调试输出，只保留扩展通知，注释掉整个 `should_print_debug` 块：

```python
# if should_print_debug:
#     try:
#         print(f"[CurriculumDebug] ...")
#     except Exception:
#         pass
```

这样只会看到扩展通知，终端输出会非常干净。
