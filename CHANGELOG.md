# Changelog

## 2025-11-08

### Added
- Documented `scripts/staged_train.py` and enhanced it with stage-specific headless controls and per-stage environment counts to support visible bridge tuning.
- Updated the staged bridge PPO config to reuse the flat policy network dimensions so Stage A checkpoints can resume without shape mismatches.

### Changed (Vision RL, branch: variant/e2e-rl_CNN)
- Sirius task (vision): tuned PPO for 256 envs — `learning_rate=2e-4`, `num_steps_per_env=32`, `num_learning_epochs=6`, `num_mini_batches=2`, `desired_kl=0.008`, `clip_param=0.15`, `entropy_coef=0.02`.
- Curriculum made conservative: smaller initial command ranges, `max_curriculum=0.6`, `curriculum_increment=0.01`, higher thresholds and minimum resets.
- Rewards strengthened for stability: `orientation=-20.0`, `base_height=-800.` to penalize tipping/falls earlier.
- Camera sync and perf: keep `obs_refresh_interval=2`, `enable_tensors=True`, `max_envs=256` to avoid rotation-induced desync.
- Vision encoder lightened: Conv channels 16/32 (was 32/64), MLP head 64-d (was 128-d) while keeping `vision_latent_dim=32`.
- Disabled noisy curriculum debug prints; kept expansion summary logs only.

### Docs
- Added EMA explanation: `EMA_EXPLAINED.md` (what EMA measures and how to read it).
- Added hyperparameter scaling guidance: `HYPERPARAMETER_SCALING.md`.
- Added before/after comparison: `HYPERPARAMETER_COMPARISON.md`.

