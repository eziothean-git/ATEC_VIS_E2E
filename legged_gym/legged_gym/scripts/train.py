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
import collections

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
            # prefer capture-on-demand to avoid rendering every simulation step
            env_cfg.camera.capture_on_demand = True
        except Exception:
            pass

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # Print camera / gym capabilities to help diagnose whether tensor (zero-copy) path is available
    try:
        has_tensor_api = hasattr(env.gym, 'get_camera_image_tensor')
        cam_tensors_enabled = bool(getattr(env_cfg, 'camera', None) and getattr(env_cfg.camera, 'enable_tensors', False))
        sim_params = getattr(env, 'sim_params', None)
        use_gpu_pipeline = getattr(sim_params, 'use_gpu_pipeline', None) if sim_params is not None else None
        print(f"[Startup] camera_tensors_enabled={cam_tensors_enabled} gym_has_get_camera_image_tensor={has_tensor_api} sim_use_gpu_pipeline={use_gpu_pipeline} num_envs={getattr(env, 'num_envs', 'N/A')}")
    except Exception:
        pass
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # prepare an in-memory camera buffer if camera is enabled
    try:
        if getattr(env_cfg, 'camera', None) is not None and getattr(env_cfg.camera, 'enable', False):
            max_mem = int(getattr(env_cfg.camera, 'max_mem_frames', 128))
            # thread-safe deque for storing recent depth frames in memory (no disk I/O)
            env._camera_frames = collections.deque(maxlen=max_mem)
            env._camera_frames_lock = threading.Lock()
    except Exception:
        pass
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

        # If running headless (no display) we use an in-memory buffer to store
        # recent depth frames instead of writing PNG/CSV/NPY to disk which is slow.
        mem_only = bool(getattr(args, 'headless', False) or getattr(env, 'headless', False))

        while not stop_event.is_set():
            try:
                # Clear previous debug lines so they are not captured in the next camera render
                try:
                    if hasattr(env, 'gym') and hasattr(env, 'viewer') and env.viewer is not None:
                        env.gym.clear_lines(env.viewer)
                except Exception:
                    pass

                # If in mem_only (headless) mode, skip triggering rendering/readback
                # because calls like get_camera_depth_images() and render_all_camera_sensors
                # can force GPU->CPU synchronization and throttle the training loop.
                if mem_only:
                    # skip capture and visualization to avoid costly GPU syncs
                    frame_idx += 1
                    stop_event.wait(sleep_interval)
                    continue

                # Safely attempt to get depth images; may raise if camera not initialized
                depth = env.get_camera_depth_images(as_torch=False)
                # Attempt to fetch the raw renderer depth buffer directly from gym for
                # comparison/debugging. Save it immediately so we can inspect pre- and
                # post-wrapper values to determine whether clipping/translation is the cause
                # of the binary output.
                raw_depth = None
                # Only attempt a direct renderer read when not in mem_only mode (it forces an extra GPU->CPU read)
                if not mem_only:
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
                    # If running in-memory-only mode (no display), push depths to env buffer
                    if mem_only and hasattr(env, '_camera_frames'):
                        try:
                            with env._camera_frames_lock:
                                env._camera_frames.append(depth.copy())
                        except Exception:
                            pass
                    else:
                        # non-mem mode: minimal save to disk to preserve previous behavior
                        try:
                            np.save(os.path.join(out_dir, 'depth_raw_latest.npy'), depth)
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

            # Note: camera position visualization has been made opt-in and is
            # intentionally not called here to avoid extra viewer/device interactions
            # that can cause errors when many envs are created (handles mismatches).

            stop_event.wait(sleep_interval)

    cam_thread = None
    if hasattr(args, 'camera_enable') and args.camera_enable:
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
