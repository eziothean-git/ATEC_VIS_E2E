Camera depth map fix (2025-11-05)

Overview
--------
This document explains a compatibility fix to the onboard camera depth handling.

Problem
-------
Some versions of the renderer return a normalized depth buffer (values in [0,1]) while others return view-space z (negative values in front of the camera). Additionally, missed rays or background pixels may appear as NaN/±Inf. When the wrapper assumed a single convention this could produce images that looked binary (near-plane vs far-plane) instead of continuous depth.

Fix
---
- Detect whether the renderer returned normalized depth (0..1) or view-space z (negative). If normalized, convert from NDC to view z using the camera near/far planes and take absolute value to get depth in meters. If view-space z, take absolute value.
- Replace non-finite values with `cfg.camera.max_depth` to preserve a stable clipping value for visualization and downstream usage.
- Produce a per-pixel hit mask indicating which pixels are valid (finite) in the renderer buffer; this mask can be returned alongside the linearized depth or saved for diagnostics.
- Added CLI flags and config options:
  - `--camera_debug_outputs` / `cfg.camera.debug_outputs` — save `depth_raw_renderer_latest.npy`, `depth_linearized_latest.npy`, `depth_mask_latest.npy` to `camera_outputs/` when enabled.
  - `--camera_mask_fraction_in_obs` / `cfg.camera.include_mask_fraction_in_obs` — append per-env fraction of valid depth pixels to the observation vector (opt-in).
  - `--camera_test_mode` / `cfg.camera.camera_test_mode` — capture a small number of frames and exit (opt-in short test mode).
  - `--camera_test_frames` / `cfg.camera.camera_test_frames` — number of frames to capture in camera test mode.

Where to look
-------------
- `legged_gym/legged_gym/envs/base/legged_robot.py` — `get_camera_depth_images` has the detection and linearization logic and an optional `return_mask` parameter.
- `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_joystick.py` — the task-specific camera wrapper has the same behavior.
- `legged_gym/legged_gym/utils/helpers.py` — CLI parsing and propagation to `env_cfg.camera`.
- `legged_gym/legged_gym/scripts/train.py` — camera monitor thread and short camera test flow; camera_test_mode is opt-in.

How to reproduce a quick test
---------------------------
1. Run a short camera test that captures frames then exits and saves debug files:

```bash
python3 legged_gym/legged_gym/scripts/train.py --task <task_name> --camera_enable --camera_test_mode --camera_test_frames 5 --camera_debug_outputs
```

2. Inspect `camera_outputs/` in the run directory for:
- `depth_raw_renderer_latest.npy` — raw renderer depth buffer
- `depth_linearized_latest.npy` — linearized depths in meters
- `depth_mask_latest.npy` — boolean mask (True for valid pixels)

Notes
-----
- The implementation currently writes debug files to the run's `camera_outputs` directory; some environment-internal paths used a repository-local folder for diagnostic stability. If you want a different location or a config-controlled path, this can be made configurable.
- If you enable `camera_mask_fraction_in_obs`, ensure any custom task `compute_observations()` logic preserves the observation ordering or explicitly appends the mask-derived scalar to match the algorithm's expected `num_observations`.

Contact
-------
If you'd like changes to the debug output layout, additional derived features (e.g., depth percentiles in obs), or a different storage location, reply here and I will update the code & docs accordingly.
