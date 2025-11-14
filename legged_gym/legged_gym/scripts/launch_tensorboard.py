#!/usr/bin/env python3
import argparse
import subprocess
import os

LOGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../logs'))
FOLDERS = [
    'sirius_curriculum',
    'sirius_teacher_curriculum',
    'sirius_flat',
]

def main():
    parser = argparse.ArgumentParser(description='快速启动 TensorBoard 查看指定实验日志')
    for folder in FOLDERS:
        parser.add_argument(f'--{folder}', action='store_true', help=f'打开 {folder} 日志')
    args = parser.parse_args()

    selected = [f for f in FOLDERS if getattr(args, f)]
    if not selected:
        print('请通过参数选择要打开的日志文件夹，例如 --sirius_diff_vis')
        return
    if len(selected) > 1:
        print('请只选择一个日志文件夹！')
        return

    base_log_dir = os.path.join(LOGS_DIR, selected[0])
    if not os.path.exists(base_log_dir):
        print(f'日志文件夹不存在: {base_log_dir}')
        return

    # 查找最新的时间戳子文件夹（按修改时间选择最新），比按名称更可靠
    subfolders = [f for f in os.listdir(base_log_dir) if os.path.isdir(os.path.join(base_log_dir, f))]
    if not subfolders:
        print(f'未找到任何时间戳日志子文件夹于: {base_log_dir}')
        return
    # 构造绝对路径并按最后修改时间选择最新目录
    subfolder_paths = [os.path.join(base_log_dir, f) for f in subfolders]
    try:
        latest_log = max(subfolder_paths, key=os.path.getmtime)
    except Exception:
        # 回退到按名称排序（保守处理），以避免未预料的异常中止
        subfolders.sort(reverse=True)
        latest_log = os.path.join(base_log_dir, subfolders[0])
    print(f'正在启动 TensorBoard，日志目录: {latest_log}')
    subprocess.run(['tensorboard', '--logdir', latest_log])

if __name__ == '__main__':
    main()
