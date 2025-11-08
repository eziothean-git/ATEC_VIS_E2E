#!/usr/bin/env python3
"""
快速测试不同配置下的显存占用和性能
"""
import subprocess
import time
import re

configs = [
    {
        "name": "256_envs_baseline",
        "envs": 256,
        "refresh": 4,
        "max_cams": 256,
    },
    {
        "name": "512_envs_moderate",
        "envs": 512,
        "refresh": 4,
        "max_cams": 256,
    },
    {
        "name": "1024_envs_aggressive",
        "envs": 1024,
        "refresh": 8,
        "max_cams": 256,
    },
    {
        "name": "2048_envs_extreme",
        "envs": 2048,
        "refresh": 8,
        "max_cams": 256,
    },
]

def get_gpu_memory():
    """获取当前 GPU 显存占用 (MB)"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip())
    except Exception:
        return -1

def run_test(config):
    """运行单个测试配置"""
    print(f"\n{'='*60}")
    print(f"Testing: {config['name']}")
    print(f"  Envs: {config['envs']}")
    print(f"  Refresh Interval: {config['refresh']}")
    print(f"  Max Cameras: {config['max_cams']}")
    print(f"{'='*60}")
    
    # 构建命令
    cmd = [
        "python", "legged_gym/scripts/train.py",
        "--task=sirius",
        f"--num_envs={config['envs']}",
        "--headless",
        "--max_iterations=5",  # 只跑 5 iterations
    ]
    
    # 记录初始显存
    mem_before = get_gpu_memory()
    print(f"GPU Memory Before: {mem_before} MB")
    
    # 运行训练
    start_time = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # 等待启动并稳定
        time.sleep(10)
        
        # 记录峰值显存
        mem_peak = get_gpu_memory()
        print(f"GPU Memory Peak: {mem_peak} MB")
        print(f"GPU Memory Delta: +{mem_peak - mem_before} MB")
        
        # 等待完成
        stdout, stderr = proc.communicate(timeout=60)
        
        # 解析 FPS
        fps_match = re.search(r"(\d+\.?\d*)\s*steps/s", stdout)
        if fps_match:
            fps = float(fps_match.group(1))
            print(f"Training FPS: {fps:.1f} steps/s")
        
    except subprocess.TimeoutExpired:
        proc.kill()
        print("TIMEOUT: Test took too long")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        elapsed = time.time() - start_time
        print(f"Test Duration: {elapsed:.1f}s")
        
        # 等待清理
        time.sleep(5)
        mem_after = get_gpu_memory()
        print(f"GPU Memory After: {mem_after} MB")

if __name__ == "__main__":
    print("Starting Memory Optimization Tests")
    print(f"Baseline GPU Memory: {get_gpu_memory()} MB\n")
    
    for config in configs:
        run_test(config)
        # 每个测试间等待 GPU 清理
        print("\nWaiting for GPU cleanup...")
        time.sleep(10)
    
    print(f"\n{'='*60}")
    print("All tests completed!")
    print(f"Final GPU Memory: {get_gpu_memory()} MB")
