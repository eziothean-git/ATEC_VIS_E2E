# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import os
from datetime import datetime
import threading
import time

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import update_cfg_from_args
import torch

def train(args):
    # load default configs so we can tweak camera settings before creating the env
    env_cfg, train_cfg_tmp = task_registry.get_cfgs(name=args.task)
    # apply CLI overrides to env_cfg
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

    # If requested, reserve an extra observation slot for depth mask fraction
    try:
        if getattr(env_cfg, 'camera', None) is not None and getattr(env_cfg.camera, 'include_mask_fraction_in_obs', False):
            env_cfg.env.num_observations = int(getattr(env_cfg.env, 'num_observations', 0)) + 1
    except Exception:
        pass

    # If camera is enabled, ensure a reasonable default position/orientation like camera_demo
    if getattr(env_cfg, 'camera', None) is not None and env_cfg.camera.enable:
        # set sensible defaults if not present
        if getattr(env_cfg.camera, 'position', None) is None:
            env_cfg.camera.position = [0.45, 0.0, -0.03]
        if getattr(env_cfg.camera, 'rpy', None) is None:
            env_cfg.camera.rpy = [0.0, 0.6, 0.0]
        # Try changing camera sensor settings that affect depth rendering.
        # These defaults are experimental diagnostics to avoid most pixels being at far plane.
        try:
            # extend far plane so we can observe larger range
            env_cfg.camera.far_plane = getattr(env_cfg.camera, 'far_plane', 10.0) * 3.0
        except Exception:
            pass
        try:
            # reduce near plane for better close-range sensitivity
            env_cfg.camera.near_plane = getattr(env_cfg.camera, 'near_plane', 0.05) * 0.2
        except Exception:
            pass
        try:
            # enable collision geometry rendering and tensor path if available
            env_cfg.camera.use_collision_geometry = True
        except Exception:
            pass
        try:
            env_cfg.camera.enable_tensors = True
        except Exception:
            pass
        # If user only wants a quick camera test, set up a short test mode here.
        # This will override some camera parameters and instruct the script to
        # capture a few frames then exit (useful for debugging depth outputs).
        try:
            # set a sensible max depth for visualization during tests
            env_cfg.camera.max_depth = float(getattr(env_cfg.camera, 'max_depth', 10.0))
            # enable collision geometry and tensor path for diagnostic rendering
            env_cfg.camera.use_collision_geometry = True
            env_cfg.camera.enable_tensors = True
            # only enable short camera_test_mode if requested via CLI/config
            if getattr(env_cfg.camera, 'camera_test_mode', False):
                env_cfg.camera.max_depth = 4.0
                env_cfg.camera.camera_test_frames = int(getattr(env_cfg.camera, 'camera_test_frames', 5))
        except Exception:
            pass

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # If camera is enabled via CLI or config, spawn a background monitor thread
    # that periodically fetches camera depth images and writes them to disk.
    stop_event = threading.Event()

    def camera_monitor():
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        out_dir = os.path.join(os.getcwd(), 'camera_outputs')
        os.makedirs(out_dir, exist_ok=True)

        frame_idx = 0
        # Try to read a reasonable sleep interval from config, fallback to 0.1s
        sleep_interval = 0.1
        try:
            # If camera_display_interval_steps is set, estimate wall time interval
            steps = int(getattr(env, 'camera_display_interval_steps', getattr(env_cfg.camera, 'display_interval_steps', 3)))
            # simulation dt available on env as dt (approx)
            sim_dt = getattr(env, 'dt', None)
            if sim_dt is None:
                sim_dt = getattr(env, 'sim_params', None)
                # leave fallback
            if sim_dt is not None and isinstance(sim_dt, (int, float)) and sim_dt > 0:
                sleep_interval = max(0.01, steps * sim_dt)
        except Exception:
            pass

        while not stop_event.is_set():
            try:
                # Clear previous debug lines so they are not captured in the next camera render
                try:
                    if hasattr(env, 'gym') and hasattr(env, 'viewer') and env.viewer is not None:
                        env.gym.clear_lines(env.viewer)
                except Exception:
                    pass

                # Safely attempt to get depth images; may raise if camera not initialized
                depth = env.get_camera_depth_images(as_torch=False)
                # Attempt to fetch the raw renderer depth buffer directly from gym for
                # comparison/debugging. Save it immediately so we can inspect pre- and
                # post-wrapper values to determine whether clipping/translation is the cause
                # of the binary output.
                raw_depth = None
                try:
                    from isaacgym import gymapi
                    raw = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handles[0], gymapi.IMAGE_DEPTH)
                    raw = np.array(raw, dtype='float32')
                    raw_depth = raw
                    # save raw renderer depth for later inspection (overwrite latest)
                    try:
                        np.save(os.path.join(out_dir, 'depth_raw_renderer_latest.npy'), raw_depth)
                    except Exception:
                        pass
                except Exception:
                    raw_depth = None
                # Save only the latest depth (overwrite previous files to avoid clutter)
                if depth is not None:
                    # overwrite wrapper-normalized depth (what get_camera_depth_images returns)
                    try:
                        np.save(os.path.join(out_dir, 'depth_raw_latest.npy'), depth)
                    except Exception:
                        pass
                    # save a visualization for the first env only and a CSV of actual depths
                    if plt is not None:
                        try:
                            import matplotlib.pyplot as _plt
                            max_depth_cfg = float(getattr(env_cfg.camera, 'max_depth', getattr(env_cfg.camera, 'far_plane', 10.0)))
                            d0 = np.abs(depth[0].astype('float32'))
                            # mask invalids
                            invalid_mask = np.isnan(d0) | np.isneginf(d0) | np.isposinf(d0)
                            d0[invalid_mask] = np.nan

                            # compute statistics on valid pixels
                            valid = ~np.isnan(d0)
                            num_valid = int(np.count_nonzero(valid))
                            total = d0.size
                            valid_frac = num_valid / total if total > 0 else 0.0
                            if num_valid > 0:
                                valid_vals = d0[valid]
                                data_min = float(np.nanmin(valid_vals))
                                data_max = float(np.nanmax(valid_vals))
                                data_mean = float(np.nanmean(valid_vals))
                            else:
                                data_min = float('nan')
                                data_max = float('nan')
                                data_mean = float('nan')

                            # choose display max: prefer actual data_max (no clipping to config) so
                            # visualization uses the true dynamic range produced by the renderer.
                            if num_valid > 0 and np.isfinite(data_max) and data_max > 0:
                                display_max = data_max
                            else:
                                display_max = max_depth_cfg

                            # prepare uint8 image: map 0..display_max -> 0..255
                            img = d0.copy()
                            img[np.isnan(img)] = display_max
                            img = np.clip(img, 0.0, display_max)
                            if display_max > 0:
                                img_u8 = (img / display_max * 255.0).astype('uint8')
                            else:
                                img_u8 = np.zeros_like(img, dtype='uint8')

                            # overwrite PNGs (latest only)
                            png_path = os.path.join(out_dir, 'depth_frame_latest.png')
                            _plt.imsave(png_path, img_u8, cmap='gray', vmin=0, vmax=255)
                            try:
                                inv = 255 - img_u8
                                png_inv = os.path.join(out_dir, 'depth_frame_latest_inverted.png')
                                _plt.imsave(png_inv, inv, cmap='gray', vmin=0, vmax=255)
                            except Exception:
                                pass
                            try:
                                if num_valid > 0:
                                    # use percentile without clamping to config max_depth
                                    vmax = float(np.percentile(valid_vals, 99))
                                else:
                                    vmax = display_max
                                if vmax <= 0:
                                    vmax = display_max
                                img_pct = img.copy()
                                img_pct = np.clip(img_pct, 0.0, vmax)
                                img_pct_u8 = (img_pct / vmax * 255.0).astype('uint8')
                                png_pct = os.path.join(out_dir, 'depth_frame_latest_pct99.png')
                                _plt.imsave(png_pct, img_pct_u8, cmap='gray', vmin=0, vmax=255)
                            except Exception:
                                pass

                            # overwrite CSV with full float depth matrix for env0
                            csv_path = os.path.join(out_dir, 'depth_matrix_env0_latest.csv')
                            try:
                                # write as floats with header
                                np.savetxt(csv_path, d0, delimiter=',', fmt='%.6f')
                            except Exception:
                                pass

                            # compute additional diagnostics: dtype, unique values (small images only), percentiles, center samples
                            try:
                                dtype = str(d0.dtype)
                                # unique values and counts (if not too many uniques)
                                uniques, counts = np.unique(d0, return_counts=True)
                                unique_info = list(zip(uniques.tolist(), counts.tolist())) if uniques.size <= 256 else [('too_many_uniques', uniques.size)]
                            except Exception:
                                dtype = 'unknown'
                                unique_info = []

                            # percentiles
                            try:
                                p10 = float(np.nanpercentile(d0, 10))
                                p50 = float(np.nanpercentile(d0, 50))
                                p90 = float(np.nanpercentile(d0, 90))
                            except Exception:
                                p10 = p50 = p90 = float('nan')

                            # center 5x5 samples
                            try:
                                h, w = d0.shape
                                ch, cw = h // 2, w // 2
                                center_samples = d0[ch-2:ch+3, cw-2:cw+3].tolist()
                            except Exception:
                                center_samples = []

                            # overwrite stats file
                            stats_path = os.path.join(out_dir, 'depth_stats_latest.txt')
                            with open(stats_path, 'w') as sf:
                                sf.write(f"frame,latest\n")
                                sf.write(f"dtype,{dtype}\n")
                                sf.write(f"valid_pixels,{num_valid}\n")
                                sf.write(f"total_pixels,{total}\n")
                                sf.write(f"valid_fraction,{valid_frac:.6f}\n")
                                sf.write(f"data_min,{data_min}\n")
                                sf.write(f"data_max,{data_max}\n")
                                sf.write(f"data_mean,{data_mean}\n")
                                sf.write(f"p10,{p10}\n")
                                sf.write(f"p50,{p50}\n")
                                sf.write(f"p90,{p90}\n")
                                sf.write(f"config_max_depth,{max_depth_cfg}\n")
                                sf.write(f"display_max_used,{display_max}\n")
                                sf.write("unique_values_and_counts,\n")
                                for u in unique_info:
                                    sf.write(f"{u[0]},{u[1]}\n")
                                sf.write("center_5x5_samples,\n")
                                for row in center_samples:
                                    sf.write(','.join(str(x) for x in row) + "\n")
                            # also write raw renderer depth diagnostics if we managed to fetch it
                            try:
                                if raw_depth is not None:
                                    rd = raw_depth
                                    rmin = float(np.nanmin(rd))
                                    rmax = float(np.nanmax(rd))
                                    # unique values (cap)
                                    rur, ruc = np.unique(rd, return_counts=True)
                                    sf.write("raw_min,%s\n" % rmin)
                                    sf.write("raw_max,%s\n" % rmax)
                                    sf.write("raw_unique_values_and_counts,\n")
                                    if rur.size <= 512:
                                        for a, b in zip(rur.tolist(), ruc.tolist()):
                                            sf.write(f"{a},{b}\n")
                                    else:
                                        sf.write(f"too_many_raw_uniques,{rur.size}\n")
                                else:
                                    sf.write("raw_unique_values_and_counts,unavailable\n")
                            except Exception:
                                sf.write("raw_unique_values_and_counts,unavailable\n")
                        except Exception:
                            pass
                    frame_idx += 1
                    # If running in short camera test mode, stop after requested frames
                    try:
                        if getattr(env_cfg.camera, 'camera_test_mode', False):
                            max_frames = int(getattr(env_cfg.camera, 'camera_test_frames', 5))
                            if frame_idx >= max_frames:
                                # create a done marker and stop the monitor
                                done_path = os.path.join(out_dir, 'camera_test_done.txt')
                                with open(done_path, 'w') as df:
                                    df.write(f"captured_frames,{frame_idx}\n")
                                stop_event.set()
                                return
                    except Exception:
                        pass
            except Exception:
                # Camera may not be initialized or rendering not available; ignore and retry
                pass

            # After capturing depth (so debug markers aren't included), redraw camera position markers
            try:
                if hasattr(env, 'visualize_camera_position'):
                    env.visualize_camera_position()
            except Exception:
                pass

            stop_event.wait(sleep_interval)

    cam_thread = None
    # Start camera monitor if:
    # 1. CLI flag --camera_enable is set, OR
    # 2. config has camera.debug_outputs = True AND not in headless mode
    should_monitor_camera = False
    if hasattr(args, 'camera_enable') and args.camera_enable:
        should_monitor_camera = True
    elif not args.headless and getattr(env_cfg, 'camera', None) is not None and getattr(env_cfg.camera, 'debug_outputs', False):
        should_monitor_camera = True
        print('[train] camera.debug_outputs=True detected (non-headless mode), starting camera monitor thread')
    
    if should_monitor_camera:
        cam_thread = threading.Thread(target=camera_monitor, daemon=True)
        cam_thread.start()
        # If camera_test_mode is active, wait for the camera monitor to finish capturing
        # the requested frames and then exit early (do not start training loop).
        try:
            if getattr(env_cfg, 'camera', None) is not None and getattr(env_cfg.camera, 'camera_test_mode', False):
                print('[train] camera_test_mode active: waiting for camera monitor to finish...')
                cam_thread.join()
                print('[train] camera test completed, exiting early.')
                return
        except Exception:
            pass

    try:
        ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    finally:
        # stop camera thread if running
        if stop_event is not None:
            stop_event.set()
        if cam_thread is not None:
            cam_thread.join(timeout=2.0)

if __name__ == '__main__':
    args = get_args()
    train(args)
