# Camera debug summary (2025-11-05, variant/e2e-rl)

Short summary:

- Problem: When enabling onboard camera during training (`--camera_enable`), depth images were binary — many pixels at 0 (near) and the remainder clipped to the configured far plane (e.g. 4.0). Training run did not produce the same visualization/save output as `camera_demo.py`.

Actions taken:

- Added a `camera_monitor` background thread in `train.py` to periodically capture and save wrapper depth outputs and RGB as `.npy`/`.png`/`.csv` plus detailed diagnostics (min/max/percentiles/unique counts/center samples).
- Added saving of the renderer's raw depth buffer as `camera_outputs/depth_raw_renderer_latest.npy` so we can distinguish renderer-level outputs from wrapper post-processing.
- Made `CameraProperties` options `use_collision_geometry` and `enable_tensors` configurable via `cfg.camera` so we can toggle render geometry and CPU/GPU tensor path for A/B tests.
- Introduced a short `camera_test_mode` in `train.py` that overrides camera parameters and automatically captures N frames then exits for quick debugging.

Key diagnostic finding:

- `depth_stats_latest.txt` (example) shows most pixels are exactly 0.0 or exactly the configured `max_depth` (4.0). Center patch is 0.0. This likely indicates either the renderer missed geometry in many pixel directions (returning -inf that gets replaced by max_depth), or the renderer returns a normalized depth buffer that the wrapper incorrectly treats as linear meters.

Next steps:

1. Run the camera test locally (environment with IsaacGym available) and upload `camera_outputs/depth_raw_renderer_latest.npy` and `depth_stats_latest.txt` so we can determine if raw depth is normalized (0..1) or already in meters.
2. If raw is normalized, convert depth-buffer values back to linear meters using near/far. If raw is already in meters but still binary, perform parameter sweep (camera pose, pitch, `use_collision_geometry` on/off) to find configuration that yields continuous depth.
3. Longer term: add asynchronous frame capture for large-scale runs and consider a raycast-based depth sampler as a renderer-independent fallback.

This file serves as the debug memory and rationale for the camera changes added in this round.
