# VARIANTS

This repository contains experimental variants of the base framework. This file records branches that are variants and documents their purpose so maintainers and contributors understand which branches are experimental and should not be merged into `master` without review.

## variant/e2e-rl
- Purpose: End-to-end reinforcement learning (E2E-RL) variant for robot-dog control. This branch contains research code, experiment configs, and training scripts that differ from the base framework.
- Branch name: `variant/e2e-rl`
- Author / Owner: ekiothean (local maintainer)
- Notes:
  - This branch is a development/experiment branch. It should not be merged into `master` automatically.
  - Large training artifacts, model checkpoints, and logs must be stored with Git LFS (the repository tracks `legged_gym/logs/sirius_two_span_bridge/**` with LFS).
  - Training logs, datasets and other ephemeral outputs should be excluded from `master` and pushed to remote storage or LFS only.
  - If you want to merge changes from this branch into `master`, open a PR and request an explicit code review and cleanup (remove large files, normalize config, ensure tests pass).

## How to use
- Development: Use `variant/e2e-rl` locally for experiments. Keep `master` stable and merge only well-reviewed changes.
- Backups: This repository keeps a backup branch `backup-before-history-clean-<TS>` which preserves pre-clean history snapshots.

## Contact
If you're not the branch maintainer and want to propose merging or rebase operations, please contact the repository maintainer via GitHub issues or the PR discussion for this branch.

---

(Automatically added by automation on behalf of branch setup.)
