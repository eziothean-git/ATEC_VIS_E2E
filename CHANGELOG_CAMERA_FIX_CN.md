# 变更记录（相机改进）

日期：2025-11-05
分支：variant/e2e-rl

概要：
- 对训练路径（非 demo）中的相机逻辑进行了重写与增强；目标是在非 headless 模式下以可配置的步率展示摄像头画面（默认每 3 步更新一帧），并暴露稳定可用的深度图接口，便于后续策略把深度图作为观测输入使用。

主要变更文件：
- `legged_gym/legged_gym/envs/base/legged_robot.py`
  - 限制实际创建摄像头的 env 数量（`cfg.camera.max_envs`），避免在大规模 env 上一次性分配过多渲染/窗口资源导致 native 崩溃。
  - 实现 `get_camera_depth_images` 与 `get_camera_rgb_images`：支持返回占位符（当未为所有 env 创建摄像头时），深度图将把 -inf 替换为 `cfg.camera.max_depth` 并裁剪到 [0, max_depth]；可选直接返回放在 env device 上的 PyTorch tensor，便于策略直接使用。
  - 实现 `_maybe_update_camera_display`：在有可见显示（$DISPLAY）且 viewer 可用时使用 OpenCV 实时显示；在无显示（如远程 SSH、headless 或容器）时把 RGB（PNG）和对应的 depth（NPY）写入 `scripts/camera_outputs/` 并在控制台打印保存路径与深度范围，以便离线检查和调试。

- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
  - 新增/公开相机配置字段：`display_interval_steps`、`display_window_name`、`max_depth`、`max_envs`（用于控制默认创建摄像头数量）。

- `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_bridge_env.py`
  - 为 diff-vis 任务默认开启相机（`enable=True`、`display_interval_steps=3`），便于示例演示和调试。

- `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_joystick.py`
  - 引入与 base 相似的相机创建/获取/显示逻辑以及 asset/刚体名的调试打印（用于诊断摄像头挂载点未命中问题）。

- `legged_gym/legged_gym/utils/helpers.py`
  - 新增 CLI 选项 `--camera_body`（命令行上覆盖相机挂载的刚体名），并在运行时将其映射到 `env_cfg.camera.body_name`。
  - 其他 camera CLI 选项（`--camera_enable`、`--camera_display_interval_steps`、`--camera_window_name`、`--camera_max_depth`、`--camera_max_envs`）用于运行时覆盖配置。

- 额外：在 `task_registry` 与 `BaseTask.__init__` 中加入了早期调试打印，用于确认初始化阶段是否到达 Python 层、打印解析后的 URDF 路径与刚体名列表。

行为说明与验证：
- 如果本地可见显示（例如在带 X11 的桌面会话），并且 viewer 可用，OpenCV 将在 `cfg.camera.display_window_name` 指定的窗口显示第一路摄像头画面（默认每 3 步更新一帧，可通过 CLI 或配置覆盖）。
- 如果运行在无 DISPLAY 的环境（例如 SSH 没有 X 转发、Wayland 或容器、或以 headless 模式运行），系统会自动把当前帧写入磁盘：
  - 保存路径：`<repo>/scripts/camera_outputs/`；文件名示例：`cam_numenvs1_step123.png`（RGB）和 `cam_numenvs1_step123.npy`（depth）。
  - 控制台会打印保存路径及深度的最小/最大值，便于核查深度正确性。
- 深度接口保证了返回值的稳定性：`get_camera_depth_images(as_torch=True)` 返回的 tensor 不含 -inf，且被裁剪到 `cfg.camera.max_depth`，并会尽量放到环境使用的 device（如 `cuda:0`），方便策略直接消费。

已知问题与缓解建议：
- 早期在为所有 env 创建摄像头的大规模跑（num_envs 大）中，曾观测到 native 层（GPU pipeline / 驱动）出现 SIGSEGV。为此增加了 `cfg.camera.max_envs` 来限制默认创建摄像头数量；建议第一次启用时把该值设为 1 以验证。若需要在大规模 env 下启用更多摄像头，可改为：
  - 只为若干 env 创建摄像头（例如第 1 个 env），或
  - 将渲染改为离线/异步抓帧并写磁盘的模式，以减少对实时渲染管线的压力。

快速验证（建议命令）：
- 在本地桌面会话（有 DISPLAY）查看窗口：

```bash
python train.py --task=sirius_diff_vis --num_envs 1 --camera_enable --camera_max_envs 1 --camera_display_interval_steps 3
```

- 在无显示会话（或容器 / SSH）查看保存的帧：

```bash
python train.py --task=sirius_diff_vis --num_envs 1 --camera_enable --camera_max_envs 1 --camera_display_interval_steps 3
# 运行后查看 scripts/camera_outputs/ 下的 png / npy 文件
```

后续改进（可选）：
- 为 `--camera_body` 提供候选提示（当 body 名不匹配时列出可选刚体名）；
- 在大规模 env 情况下实现异步抓帧并写磁盘以降低对渲染/驱动的压力；
- 如果把摄像头作为策略观测的一部分，建议在数据管线中加入深度预处理（例如下采样、归一化或压缩），减少带宽和显存使用。

---

以上为本次相机改动的中文变更说明，若需我把该内容追加到现有 `CHANGELOG_CAMERA_FIX.md` 文件的末尾而不是新建文件，我可以继续修改（需要覆盖原文件）。
