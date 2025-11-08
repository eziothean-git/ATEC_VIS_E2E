#!/usr/bin/env python3
"""
Simple staged training helper:
- Stage A: train on a (flat) task (default: 'sirius') for N iterations and save run
- Stage B: resume from Stage A's latest checkpoint and continue training on bridge task (default: 'sirius_two_span_bridge')

Usage (basic):
    python scripts/staged_train.py --stage_a_iters 800 --stage_b_iters 400 --num_envs 64

The script calls the repository's `legged_gym/legged_gym/scripts/train.py` and looks
for the latest run folder that contains the stage A run_name, then resumes from its
latest checkpoint for stage B.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, env=None):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
    # stream output
    try:
        for line in proc.stdout:
            print(line, end='')
    except KeyboardInterrupt:
        proc.kill()
        raise
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")
    return rc


def find_latest_run(logs_root: Path, run_name_substr: str):
    """Find the most recently modified run directory under logs_root whose name contains run_name_substr."""
    if not logs_root.exists():
        return None
    candidates = []
    for root, dirs, files in os.walk(logs_root):
        for d in dirs:
            if run_name_substr in d:
                cand = Path(root) / d
                candidates.append(cand)
    if not candidates:
        return None
    # choose by modification time
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_model(run_dir: Path):
    """Find latest model_*.pt file in run_dir (recursive). Returns absolute path or None."""
    models = list(run_dir.rglob('model_*.pt'))
    if not models:
        return None
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage_a_task', type=str, default='sirius', help='Stage A task name')
    parser.add_argument('--stage_b_task', type=str, default='sirius_two_span_bridge', help='Stage B task name')
    parser.add_argument('--stage_a_iters', type=int, default=800, help='Iterations for stage A')
    parser.add_argument('--stage_b_iters', type=int, default=400, help='Iterations for stage B')
    parser.add_argument('--num_envs', type=int, default=64, help='Number of envs to run for both stages (overridden by stage-specific values)')
    parser.add_argument('--stage_a_num_envs', type=int, default=None, help='Number of envs for stage A (defaults to --num_envs)')
    parser.add_argument('--stage_b_num_envs', type=int, default=None, help='Number of envs for stage B (defaults to --num_envs)')
    parser.add_argument('--run_base', type=str, default='staged', help='Base run name prefix')
    parser.add_argument('--headless', action='store_true', help='Pass --headless to both stages (overridden by stage-specific flags)')
    parser.add_argument('--stage_a_headless', action='store_true', help='Run Stage A in headless mode (overrides global)')
    parser.add_argument('--stage_b_headless', action='store_true', help='Run Stage B in headless mode (overrides global)')
    parser.add_argument('--stage_a_show', action='store_true', help='Force Stage A to render (viewer on) regardless of headless flags')
    parser.add_argument('--stage_b_show', action='store_true', help='Force Stage B to render (viewer on) regardless of headless flags')
    parser.add_argument('--train_script', type=str, default='legged_gym/legged_gym/scripts/train.py', help='Path to train.py')
    parser.add_argument('--logs_dir', type=str, default='legged_gym/logs', help='Root logs directory (relative or absolute)')
    parser.add_argument('--checkpoint', type=int, default=-1, help='Checkpoint to load for stage B (-1 = latest)')
    parser.add_argument('--extra_args', type=str, default='', help='Extra args appended to train.py commands (quoted)')
    parser.add_argument('--dry_run', action='store_true', help='If set: only print resolved paths and commands, do not execute training')
    args = parser.parse_args()

    cwd = Path.cwd()
    # Try several locations for train.py so the helper works regardless of current cwd.
    # Order of preference:
    #  1) cwd / args.train_script (what user provided, or default relative to cwd)
    #  2) script_dir.parent / args.train_script (repo-root relative path)
    #  3) script_dir / args.train_script (script-local relative)
    #  4) search repo_root for a matching train.py (last resort)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    train_py = cwd / args.train_script
    if not train_py.exists():
        alt = repo_root / args.train_script
        if alt.exists():
            train_py = alt
        else:
            alt2 = script_dir / args.train_script
            if alt2.exists():
                train_py = alt2
            else:
                # Search for train.py under repo_root and try to pick a sensible candidate
                candidates = list(repo_root.rglob('train.py'))
                chosen = None
                for c in candidates:
                    # prefer legged_gym path if present
                    if 'legged_gym' in str(c):
                        chosen = c
                        break
                if chosen is None and candidates:
                    chosen = candidates[0]
                if chosen is not None:
                    train_py = chosen

    if not train_py.exists():
        print(f"train.py not found at any candidate path. Last tried: {train_py}")
        print('Checked cwd, script dir, repo root and performed a recursive search under repo root.')
        sys.exit(2)
    else:
        print('Using train.py at:', train_py)

    # Compute legged gym root (two levels above train.py: .../legged_gym)
    legged_gym_root = train_py.resolve().parents[2]

    stage_a_envs = args.stage_a_num_envs if args.stage_a_num_envs is not None else args.num_envs
    stage_b_envs = args.stage_b_num_envs if args.stage_b_num_envs is not None else args.num_envs

    stage_a_headless = args.headless
    stage_b_headless = args.headless
    if args.stage_a_headless:
        stage_a_headless = True
    if args.stage_b_headless:
        stage_b_headless = True
    if args.stage_a_show:
        stage_a_headless = False
    if args.stage_b_show:
        stage_b_headless = False

    # Build Stage A command (but defer execution; support dry_run)
    run_name_a = f"{args.run_base}_flat"
    cmd_a = [sys.executable, str(train_py), '--task', args.stage_a_task, '--run_name', run_name_a, '--max_iterations', str(args.stage_a_iters), '--num_envs', str(stage_a_envs)]
    if stage_a_headless:
        cmd_a += ['--headless']
    if args.extra_args:
        cmd_a += args.extra_args.split()

    # Build Stage B command as well (for dry-run printing)
    run_name_b = f"{args.run_base}_bridge"
    cmd_b = [sys.executable, str(train_py), '--task', args.stage_b_task, '--resume', '--load_run', '<<RUN_DIR>>', '--checkpoint', str(args.checkpoint), '--run_name', run_name_b, '--max_iterations', str(args.stage_b_iters), '--num_envs', str(stage_b_envs)]
    if stage_b_headless:
        cmd_b += ['--headless']
    if args.extra_args:
        cmd_b += args.extra_args.split()

    print('\nResolved train.py ->', train_py)
    # resolve logs_root: prefer legged_gym_root-relative path for relative args.logs_dir
    logs_arg = Path(args.logs_dir)
    logs_candidates = []
    if logs_arg.is_absolute():
        logs_root = logs_arg
        logs_candidates.append(logs_root)
    else:
        logs_root = repo_root / logs_arg
        logs_candidates.append(logs_root)
        logs_candidates.append(legged_gym_root / logs_arg)
        logs_candidates.append(script_dir / logs_arg)
        logs_candidates.append(cwd / logs_arg)
        logs_candidates.append(legged_gym_root / 'logs')
        logs_candidates.append(repo_root / 'logs')
        logs_candidates.append(script_dir / 'logs')
        logs_candidates.append(cwd / 'logs')

    # ensure unique absolute candidates (ignore None)
    unique_logs_candidates = []
    seen = set()
    for cand in logs_candidates:
        if cand is None:
            continue
        cand_abs = cand if cand.is_absolute() else cand.resolve()
        key = str(cand_abs)
        if key not in seen:
            seen.add(key)
            unique_logs_candidates.append(cand_abs)

    if not unique_logs_candidates:
        unique_logs_candidates.append(logs_root)

    preferred_logs = unique_logs_candidates[0]
    print('Resolved logs root (preferred) ->', preferred_logs)
    if len(unique_logs_candidates) > 1:
        print('Additional logs search locations:')
        for cand in unique_logs_candidates[1:]:
            print('  -', cand)
    print('\nStage A command:')
    print(' '.join(map(str, cmd_a)))
    print('\nStage B command (placeholder for run dir):')
    print(' '.join(map(str, cmd_b)).replace('<<RUN_DIR>>', '<stageA_run_dir>'))

    if args.dry_run:
        print('\nDry run requested: not executing training. Exiting.')
        return

    print('\n=== Stage A: training on', args.stage_a_task, 'for', args.stage_a_iters, 'iterations ===')
    run_cmd(cmd_a)

    run_dir = None
    for candidate in unique_logs_candidates:
        print('\nLooking for Stage A run directory under', candidate)
        if candidate.exists():
            run_dir = find_latest_run(candidate, run_name_a)
            if run_dir is not None:
                break
    if run_dir is None:
        print('ERROR: Could not find run directory for stage A with name containing', run_name_a)
        print('Searched the following locations:')
        for cand in unique_logs_candidates:
            print('  -', cand)
        sys.exit(3)
    print('Found Stage A run dir:', run_dir)

    # locate model checkpoint
    model_path = find_latest_model(run_dir)
    if model_path is None:
        print('WARNING: No model_*.pt found inside', run_dir, 'will attempt to resume using --load_run with checkpoint -1')
    else:
        print('Found latest checkpoint:', model_path)

    # Stage B: resume on bridge
    run_name_b = f"{args.run_base}_bridge"
    cmd_b = [sys.executable, str(train_py), '--task', args.stage_b_task, '--resume', '--load_run', str(run_dir), '--checkpoint', str(args.checkpoint), '--run_name', run_name_b, '--max_iterations', str(args.stage_b_iters), '--num_envs', str(stage_b_envs)]
    if stage_b_headless:
        cmd_b += ['--headless']
    if args.extra_args:
        cmd_b += args.extra_args.split()

    print('\n=== Stage B: resuming on', args.stage_b_task, 'for', args.stage_b_iters, 'iterations ===')
    print('Resuming from run dir:', run_dir)
    if model_path is not None:
        print('Requested checkpoint arg (train.py will resolve):', args.checkpoint, '(use -1 for latest)')
    run_cmd(cmd_b)

    print('\nStaged training finished.')


if __name__ == '__main__':
    main()
