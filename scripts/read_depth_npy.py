#!/usr/bin/env python3
import os
import numpy as np
import sys
p = '/home/eziothean/Sirius_RL_Gym-master/legged_gym/legged_gym/scripts/camera_outputs/depth_raw_latest.npy'
if len(sys.argv) > 1:
    p = sys.argv[1]
if not os.path.exists(p):
    print(f"File not found: {p}")
    sys.exit(2)
try:
    arr = np.load(p, allow_pickle=False)
except Exception as e:
    print(f"Failed to load npy: {e}")
    raise
# If stacked (num_envs, H, W) pick first env for visualization
if arr.ndim == 3:
    arr0 = arr[0]
else:
    arr0 = arr
arrf = arr0.astype('float32')
# stats
finite_mask = np.isfinite(arrf)
nan_count = int(np.count_nonzero(~finite_mask))
finite_vals = arrf[finite_mask]
print(f"Loaded: {p}")
print(f"shape: {arr.shape}, dtype: {arr.dtype}")
print(f"nan_or_inf count: {nan_count} / {arrf.size}")
if finite_vals.size > 0:
    mn = float(np.min(finite_vals))
    mx = float(np.max(finite_vals))
    mean = float(np.mean(finite_vals))
    med = float(np.median(finite_vals))
    p1 = float(np.percentile(finite_vals, 1))
    p50 = float(np.percentile(finite_vals, 50))
    p99 = float(np.percentile(finite_vals, 99))
    print(f"finite depth stats (meters): min={mn:.6f}, max={mx:.6f}, mean={mean:.6f}, median={med:.6f}")
    print(f"percentiles: p1={p1:.6f}, p50={p50:.6f}, p99={p99:.6f}")
    # unique values heuristic
    unique_vals, counts = np.unique(np.round(finite_vals.flatten(), 6), return_counts=True)
    if unique_vals.size <= 20:
        print("unique values (<=20):")
        for v,c in zip(unique_vals, counts):
            print(f"  {v}: {c}")
    else:
        print(f"unique values: {unique_vals.size} values (too many to list)")
    # fraction near zero or equal to common far value
    zero_frac = float(np.count_nonzero(np.isclose(finite_vals, 0.0)))/finite_vals.size
    print(f"fraction == 0.0 (approx): {zero_frac*100:.3f}%")
else:
    print("No finite depth values found.")
# try to save a visualization if matplotlib is available
out_dir = os.path.dirname(p)
vis_path = os.path.join(out_dir, 'depth_raw_latest_vis.png')
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cmap = 'turbo' if 'turbo' in plt.colormaps() else 'viridis'
    # prepare display: mask inf/nan to transparent and clip to p99
    disp = arrf.copy()
    disp[~np.isfinite(disp)] = np.nan
    clip_max = np.nanpercentile(disp, 99)
    if not np.isfinite(clip_max) or clip_max <= 0:
        clip_max = np.nanmax(disp)
    plt.figure(figsize=(6,6))
    im = plt.imshow(disp, cmap=cmap, vmin=0, vmax=clip_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth (meters) - visualized (p99 clip)')
    plt.axis('off')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {vis_path}")
except Exception as e:
    print(f"Could not create visualization (matplotlib missing or error): {e}")
# Also save a small sampled CSV for quick inspection
csv_path = os.path.join(out_dir, 'depth_raw_latest_sampled.csv')
try:
    samp = arrf
    # downsample to at most 100x100 for CSV
    h,w = samp.shape
    sh = min(h, 100)
    sw = min(w, 100)
    idx_h = (np.linspace(0, h-1, sh)).astype(int)
    idx_w = (np.linspace(0, w-1, sw)).astype(int)
    small = samp[np.ix_(idx_h, idx_w)]
    np.savetxt(csv_path, small, delimiter=',', fmt='%.6f')
    print(f"Saved sampled CSV: {csv_path}")
except Exception as e:
    print(f"Could not save sampled CSV: {e}")
print("Done.")
