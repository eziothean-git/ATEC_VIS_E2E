# 变更记录（相机改进）

日期：2025-11-05
分支：variant/e2e-rl

概要：
- 对训练路径（非 demo）中的相机逻辑进行了重写与增强；目标是在非 headless 模式下以可配置的步率展示摄像头画面（默认每 3 步更新一帧），并暴露稳定可用的深度图接口，便于后续策略把深度图作为观测输入使用。
````markdown
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
   # 变更记录：相机深度线性化修复与诊断支持（中文）

   日期：2025-11-05
   分支：variant/e2e-rl

   概述：
   - 修复训练路径中摄像头深度仅表现为“近/远两档（near/far）”的二值化问题。
     根因是不同版本/配置的渲染器返回的深度语义不一致（有时是归一化 depth-buffer（0..1），有时直接返回 view-space z（通常为负值）且包含 NaN/Inf 表示未命中）。
   - 本次改动在 wrapper 层实现自动检测并线性化为“米”单位的深度，同时保留 hit-mask（哪些像素为有效命中）。另外添加了可选的调试输出（保存 renderer 原始缓冲、线性化结果与掩码）和若干 CLI 开关，便于离线诊断与把掩码信息加入观测向量。

   主要变更文件（摘要）：
   - `legged_gym/legged_gym/envs/base/legged_robot.py`
     - 在 `get_camera_depth_images` 中实现：
       - 自动检测 renderer 返回的深度语义（归一化 0..1 vs view-space z）并进行线性化（NDC->z 或取绝对值），输出单位为米；
       - 生成 hit-mask（有效像素），并用 `cfg.camera.max_depth` 替换非有限值（NaN/±Inf）；
       - 支持 `as_torch=True` 返回位于环境 device 上的 Tensor，支持 `return_mask=True` 一并返回掩码；
       - 可选将调试产物写入磁盘（由 `cfg.camera.debug_outputs` 控制）。
   - `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_joystick.py`
     - 同步实现深度线性化与掩码支持，保证任务特定代码能使用一致的深度语义。
   - `legged_gym/legged_gym/utils/helpers.py`
     - 新增并映射若干 CLI 选项：
       - `--camera_debug_outputs`：保存 renderer 原始深度、线性化深度与掩码至 `scripts/camera_outputs/`；
       - `--camera_mask_fraction_in_obs`：在观测向量末端追加每个 env 的有效深度像素比例（占位已在 train 脚本中保留）；
       - `--camera_test_mode` / `--camera_test_frames`：短时捕获若干帧并退出，便于快速诊断。
   - `legged_gym/legged_gym/scripts/train.py`
     - 增加了 `camera_monitor` 后台线程：周期性获取深度并保存可视化（PNG）、数值矩阵（CSV/NPY）和统计（min/max/percentiles/unique counts）；当 `--camera_test_mode` 启用时，捕获指定帧数后退出（短测模式）；否则训练按常规流程运行（不再强制短测模式）。

   如何复现与使用（示例）：
   - 快速相机诊断（捕获数帧并保存调试文件）：

   ```bash
   python3 train.py --task=sirius_diff_vis --num_envs 1 --camera_enable --camera_test_mode --camera_test_frames 5 --camera_debug_outputs
   ```

   - 正常训练同时启用摄像头（不会提前退出）：

   ```bash
   python3 train.py --task=sirius_diff_vis --num_envs 1 --camera_enable
   ```

   - 将掩码有效像素比例追加到观测向量（会在训练入口处为 obs 大小保留一个 slot）：

   ```bash
   python3 train.py --task=sirius_diff_vis --num_envs 1 --camera_enable --camera_mask_fraction_in_obs
   ```

   调试输出位置说明：
   - 调试文件默认保存到仓库下 `legged_gym/legged_gym/scripts/camera_outputs/`（`train.py` 的 `camera_monitor` 也会写到运行目录下的 `camera_outputs/`）。保存文件示例：
     - `depth_raw_renderer_latest.npy`：renderer 原始深度缓冲（可能为负的 view-space z 或 0..1 的归一化值）；
     - `depth_linearized_latest.npy`：线性化为米的深度（连续值，NaN/Inf 已由 `max_depth` 替代）；
     - `depth_mask_latest.npy`：命中掩码（True=有效命中）。

   注意事项与建议：
   - 不同选项与 renderer/driver 的组合在大规模 env（num_envs 很大）上可能触发 native 层不稳定（GPU/驱动层面）；建议首次启用相机时将 `--camera_max_envs 1` 或 `--num_envs 1` 以最小化风险。若需要为更多 env 启用摄像头，考虑异步写盘或仅为部分 env 创建摄像头。

   如需我把调试输出路径统一到运行目录、或把掩码比例在特定任务的 `compute_observations()` 里显式拼接（而不是在基础类中 append），我可以继续修改并添加小单元测试以保证维度匹配。

   ---

   （此文件已由脚本自动更新，提交并推送到当前分支。若需要把该变更合并到其它分支或开 PR，我也可以为你创建 PR。）
