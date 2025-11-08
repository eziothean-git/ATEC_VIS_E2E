#!/usr/bin/env python3
import numpy as np
import os
p = '/home/eziothean/Sirius_RL_Gym-master/legged_gym/legged_gym/scripts/camera_outputs/depth_raw_renderer_latest.npy'
if not os.path.exists(p):
    print('Renderer raw file not found:', p)
    raise SystemExit(2)
arr_raw = np.load(p, allow_pickle=False)
# ensure batch axis
if arr_raw.ndim == 2:
    arr_raw = arr_raw[np.newaxis, ...]

arr_linear = arr_raw.copy().astype('float32')
hit_mask = np.isfinite(arr_raw)
try:
    arr_max = float(np.nanmax(arr_raw))
    arr_min = float(np.nanmin(arr_raw))
except Exception:
    arr_max = 1.0
    arr_min = 0.0

max_depth = 4.0
# detect normalized
if arr_max <= 1.01 and arr_min >= -0.01:
    near = 0.05
    far = 10.0
    ndc = arr_raw * 2.0 - 1.0
    denom = (far + near - ndc * (far - near))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (2.0 * near * far) / denom
    arr_linear = np.abs(z.astype('float32'))
else:
    with np.errstate(invalid='ignore'):
        arr_linear = np.abs(arr_raw)

arr_linear[~np.isfinite(arr_linear)] = max_depth
arr_linear = np.clip(arr_linear, 0.0, max_depth)

out_dir = os.path.dirname(p)
np.save(os.path.join(out_dir, 'depth_linearized_latest.npy'), arr_linear[0] if arr_linear.shape[0]==1 else arr_linear)
np.save(os.path.join(out_dir, 'depth_mask_latest.npy'), hit_mask[0] if hit_mask.shape[0]==1 else hit_mask)
print('Saved depth_linearized_latest.npy and depth_mask_latest.npy to', out_dir)
