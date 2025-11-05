# Sirius_RL_Gym · ATEC2025 吊桥穿越扩展说明

本目录是在 **原始 Sirius_RL_Gym 仓库** 基础上的一组补丁与新环境，用于复现 / 训练 ATEC2025 线下赛任务四「吊桥穿越」的强化学习策略。

主要内容包括：

- 自定义 **吊桥地形**（两段桥面 + 休息区），参数对齐 ATEC 场地尺寸；
- 新增可视化调试任务 `sirius_diff_vis`；
- 若干 bugfix（初始位置、地形加载、Isaac Gym 动态库路径等）；

> ⚠️ 本目录不包含 Isaac Gym、rsl_rl 等依赖，只包含在原始仓库基础上的增量修改。要运行训练代码，请先准备原始仓库，再把本目录内容 merge 进去。

---

## 1. 依赖环境

- **操作系统**：Ubuntu 20.04 / 22.04（推荐）
- **Python**：3.8
- **PyTorch**：1.13.0 + cu116
- **Isaac Gym**：Preview 4  
  安装目录类似：`/home/<user>/IsaacGym_Preview_4_Package/`
- **rsl_rl**：从官方仓库安装（PPO 实现）
- **CUDA**：11.x，与上面 PyTorch 版本匹配

### 1.1 Isaac Gym & 动态库路径

运行时若遇到：
```text
ImportError: libpython3.8m.so.1.0: cannot open shared object file
```
可使用：
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```
若需永久生效：
```bash
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib' >> ~/.bashrc
source ~/.bashrc
```

---

## 2. 项目目录结构

```text
Sirius_RL_Gym/
├─ legged_gym/
│  ├─ scripts/
│  │   ├─ train.py                 # 训练入口
│  │   └─ play.py                  # 策略回放入口
│  │
│  ├─ legged_gym/
│  │   ├─ envs/
│  │   │   ├─ sirius_diff_vis/     # ✅ 新增：吊桥可视化 / 调试环境
│  │   │   │   ├─ __init__.py
│  │   │   │   ├─ sirius_joystick.py
│  │   │   │   └─ sirius_diff_vis_config.py
│  │   │   │
│  │   │   └─ __init__.py          # ✅ 修改：注册新 task
│  │   │
│  │   ├─ utils/
│  │   │   ├─ bridge_terrain.py    # ✅ 新增：吊桥地形构造脚本
│  │   │   └─ ...
│  │   │
│  │   └─ ...
│  │
│  └─ ...
└─ README_ATEC2025.md
```

> 当前版本仅包含 `sirius_diff_vis` 环境，不再区分 release/vis 版本。

---

## 3. 合并方式

假设：
- 原始仓库路径：`~/code/Sirius_RL_Gym`
- 当前 ATEC 扩展路径：`~/code/Sirius_RL_Gym_ATEC_patch`

执行：
```bash
cd ~/code/Sirius_RL_Gym
cp -r ~/code/Sirius_RL_Gym_ATEC_patch/* .
```
或仅复制以下部分：
```bash
cp -r ~/code/Sirius_RL_Gym_ATEC_patch/legged_gym/legged_gym/envs/sirius_diff_vis ./legged_gym/legged_gym/envs/
cp -r ~/code/Sirius_RL_Gym_ATEC_patch/legged_gym/legged_gym/utils/bridge_terrain.py ./legged_gym/legged_gym/utils/
```

确认 `envs/__init__.py` 中有：
```python
task_registry.register("sirius_diff_vis", SiriusDiffVisEnv, SiriusDiffVisCfg, SiriusDiffVisCfgPPO)
```

---

## 4. 吊桥场景配置

桥面参数：
- 桥高：1.0 m
- 平台（出发/中段/终点）：长 1.0 m，宽 0.75 m
- 第一段桥面：板宽 0.25 m，板间距 0.05 m，总长 3.05 m
- 第二段桥面：板宽 0.25 m，板间距 0.15 m，总长 4.15 m

当前版本在 `bridge_env` 中仅创建 **一个环境实例**，用于方便可视化与验证目标速度方向是否正确。

---

## 5. 启动训练

在激活环境后进入脚本目录：
```bash
conda activate sirius2
cd legged_gym/scripts
python train.py --task=sirius_diff_vis
```

说明：
- 启动后自动加载吊桥地形并渲染；
- 机器人出生点：起始平台上表面中心半径 0.25 m 内随机采样；
- 可按 `v` 开关渲染查看目标方向；
- 目前环境数量为 1，便于验证目标矢量方向是否正确。

---

## 6. Change Notes

### 2025-10-31
- 新增 `bridge_terrain.py`，可生成两段桥面 + 平台。
- 新增 `sirius_diff_vis` 环境用于可视化。

### 2025-11-01 ~ 11-02
- 完成 `sirius_diff_vis` 环境逻辑；支持地形渲染与初始位姿设定。

### 2025-11-03
- 更新 task 注册逻辑，修复导入错误。
- 新增文档说明文件。

### 2025-11-04
- 修复 Actor 创建报错。
- 初始位置调整至出发区平台中心。
- 渲染模式从线框切换至实体显示。
- 环境数量暂设为 1 以便调试目标速度。

---

## 7. 常见问题

### Q1. 环境显示为线框？
在 Isaac Gym Viewer 中切换渲染模式，或在 config 文件中修改 viewer 参数。

### Q2. 无法加载任务？
确认 `task_registry` 中存在 `sirius_diff_vis` 的注册项。
