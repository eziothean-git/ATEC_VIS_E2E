# Changelog: Camera visualization fix

Date: 2025-11-05
Branch: variant/e2e-rl

Summary:
- Fix: `visualize_camera_position` in `legged_gym/envs/base/legged_robot.py` no longer calls the GPU-incompatible API `get_actor_rigid_body_states` after simulation start. It now uses the tensor API (`self.root_states`) to obtain base positions for visualization, avoiding an IndexError when running with the GPU pipeline.

Details:
- Problem: On GPU pipeline, `GymGetActorRigidBodyStates` is not available after simulation starts. Code attempted to index into the returned body state and raised `IndexError: index 0 is out of bounds for axis 0 with size 0`.
- Resolution: Replaced the incompatible call with a tensor-API based read of `self.root_states[:, :3]` (refreshed via `refresh_actor_root_state_tensor`) and used base position + configured camera offset to draw visualization markers. This is compatible with GPU pipeline and avoids the IndexError.

Notes & next steps:
- Current implementation approximates camera world pose using the robot base/root position plus camera local offset. If you need exact body-level attachment pose (for attachments on non-root bodies), we can implement a more precise mapping using the rigid-body state tensor and actor/body index mapping.
- The change is small and low-risk. Please run `python legged_gym/scripts/camera_demo.py` in your environment to verify visual markers show and no IndexError occurs.

Commit: fix(camera): use tensor API in visualize_camera_position to avoid GPU pipeline get_actor_rigid_body_states

If you want the precise body-level position fix, reply and I'll implement retrieving rigid body state tensor mapping to actor/body indices.

- Fix: Auto-linearize renderer depth (NDC or view-space z) to meters, preserve hit-mask, add CLI flags for debug outputs and mask-in-obs.
  - Files changed: `legged_gym/legged_gym/envs/base/legged_robot.py`, `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_joystick.py`, `legged_gym/legged_gym/utils/helpers.py`, `legged_gym/legged_gym/scripts/train.py`
  - How to reproduce: run train with `--camera_enable --camera_test_mode --camera_debug_outputs`.

- Chore: Make periodic camera performance logging opt-in via `cfg.camera.print_perf` (default: False).
  - Rationale: avoid noisy logs during normal runs; allow turning on per-run performance summaries for debugging.
  - Files changed: `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_joystick.py`, `legged_gym/legged_gym/envs/sirius_diff_vis/sirius_flat_config.py`, `README.md`, `docs/CAMERA_FIX.md`.
  - How to enable: set `cfg.camera.print_perf = True` in your config (or add `print_perf: true` under `camera:` in YAML) before creating the env.
