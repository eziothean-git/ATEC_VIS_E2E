# 性能调试指南

## 问题现象
在非 headless 模式下，每过一段时间（打印完训练进度后）会出现明显卡顿，导致无法流畅观察模型表现。

## 已修复的问题

### 1. **重复调用 `post_physics_step()` (Critical Bug)**
**问题**：在 `step()` 函数中，`post_physics_step()` 被调用了两次：
- 第 147 行：计算观测和奖励
- 第 157 行：重复调用（多余）

**影响**：
- 所有观测计算执行 2 次
- 所有奖励计算执行 2 次  
- **深度相机图像获取执行 2 次**（显存访问开销巨大）
- 性能直接下降 ~50%

**修复**：移除第 157 行的重复调用

### 2. **添加详细性能监控**
现在每 500 步会自动打印性能分析（仅非 headless 模式）：

```
[Performance] num_envs=256, steps=500
  render:        12.34ms (45.2%)
  simulate:       8.56ms (31.3%)
  set_dof:        2.11ms ( 7.7%)
  refresh_dof:    1.23ms ( 4.5%)
  post_physics:   3.05ms (11.2%)
  camera_fetch:  15.60ms (called 167x, 57.1% amortized)
  TOTAL:         27.29ms/step (36.6 FPS)
```

**解读**：
- `render`: 主要瓶颈，包含 `gym.step_graphics()` + `gym.draw_viewer()`
- `camera_fetch`: 深度相机获取耗时（已优化为每 3 步刷新一次）
- 百分比显示各部分占比，帮助定位瓶颈

## 性能瓶颈分析

### 典型瓶颈排序（非 headless 模式）

1. **图形渲染** (40-60%)
   - `gym.fetch_results()` - GPU → CPU 同步
   - `gym.step_graphics()` - 渲染管线更新
   - `gym.draw_viewer()` - 窗口绘制
   - `gym.sync_frame_time()` - 帧同步等待

2. **深度相机** (10-30%)
   - `get_camera_depth_images()` - GPU 深度缓冲读取
   - 已通过刷新间隔优化（每 3 步刷新）
   - 已通过相机数量限制优化（最多 512 个）

3. **物理仿真** (20-35%)
   - `gym.simulate()` - PhysX 计算
   - 环境数量越多，耗时越高

4. **观测/奖励计算** (5-15%)
   - `post_physics_step()` - Python/PyTorch 计算
   - 复杂奖励函数会增加开销

## 优化建议

### 立即可用的优化

1. **减少渲染频率**（如果不需要实时观察）：
   ```python
   # 在 base_task.py 的 render() 中添加跳帧
   if self.common_step_counter % 5 != 0:  # 每 5 步渲染 1 次
       return
   ```

2. **减少环境数量**（非 headless 时）：
   ```bash
   python legged_gym/scripts/train.py --task=sirius --num_envs=256
   ```
   推荐：256-512 个环境用于可视化训练，4096 用于 headless 最终训练

3. **禁用调试可视化**：
   ```python
   # sirius_flat_config.py
   class env:
       debug_viz = False  # 禁用额外的调试绘制
   ```

4. **使用 headless 模式**（最快）：
   ```bash
   python legged_gym/scripts/train.py --task=sirius --num_envs=4096 --headless
   ```

### 相机优化（已实现）

- ✅ 刷新间隔：`obs_refresh_interval = 3`（每 3 步获取一次深度图）
- ✅ 相机限制：`max_envs = 512`（最多 512 个物理相机）
- ✅ 轮转机制：512 个相机在 4096 个环境间轮转，保证所有环境最终都能获取深度数据

### 高级优化（需要修改代码）

1. **异步渲染**：
   ```python
   # 在 render() 中禁用同步
   self.gym.sync_frame_time(self.sim)  # 注释掉这行
   ```
   代价：可能出现画面撕裂

2. **降低图形质量**：
   - 减少 MSAA 采样数
   - 降低阴影质量
   - 禁用后处理效果

3. **专用观察窗口**：
   只渲染少数几个环境（例如前 4 个），其余环境不创建可视化

## 测试流程

### 1. 运行性能测试
```bash
cd /home/eziothean/ATEC_VIS_E2E
source .venv/bin/activate.fish

# 非 headless 模式，观察性能输出
python legged_gym/scripts/train.py --task=sirius --num_envs=256
```

### 2. 观察性能输出
每 500 步会自动打印：
- 如果 `render` 占比 > 50%：主要瓶颈是图形渲染
- 如果 `camera_fetch` 占比 > 30%：考虑增加刷新间隔
- 如果 `simulate` 占比 > 40%：物理仿真是瓶颈，减少环境数

### 3. 对比测试
```bash
# Headless 模式（无渲染开销）
python legged_gym/scripts/train.py --task=sirius --num_envs=1024 --headless

# 检查 GPU 显存
watch -n 1 nvidia-smi
```

## 预期性能（RTX 3090）

| 模式 | 环境数 | FPS | 显存占用 | 备注 |
|------|--------|-----|----------|------|
| Headless | 4096 | ~120 | 12-15 GB | 最快训练 |
| Headless | 1024 | ~200 | 6-8 GB | 快速训练 |
| Viewer | 512 | ~40 | 8-10 GB | 可视化训练 |
| Viewer | 256 | ~60 | 5-7 GB | 流畅观察 |
| Viewer | 64 | ~100 | 3-4 GB | 超流畅 |

## 常见问题

### Q: 为什么打印进度后会卡一下？
A: PPO 算法在收集完一批经验后会进行策略更新（反向传播），这会短暂占用 GPU。这是正常现象，不影响训练效果。

### Q: 卡顿时 GPU 利用率很低？
A: 可能是 CPU → GPU 数据传输瓶颈，或者是 `fetch_results()` 同步等待。尝试增加 `decimation` 参数减少控制频率。

### Q: 修复后还是很卡？
A: 检查：
1. 其他程序是否占用 GPU（`nvidia-smi`）
2. 系统是否在录屏或截图
3. 显示器刷新率设置（建议 60Hz）
4. Isaac Gym 窗口是否全屏（全屏性能更好）

### Q: Headless 模式下如何观察训练？
A: 可以：
1. 定期保存模型并在非 headless 模式下 play
2. 使用 TensorBoard 监控奖励曲线
3. 录制视频：`python legged_gym/scripts/play.py --task=sirius --load_run=xxx`

## 下一步

修复后建议：
1. 先在非 headless 模式下测试 256 envs，确认模型行为正常
2. 切换到 headless 模式，使用 4096 envs 进行完整训练
3. 定期（每 100 iterations）保存 checkpoint
4. 训练完成后用 play.py 可视化最终效果
