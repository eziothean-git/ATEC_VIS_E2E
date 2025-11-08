# 相机调试会话纪要（2025-11-05，variant/e2e-rl）

简短摘要：
- 问题：在训练流程启用相机（`--camera_enable`）时，深度图像表现为“二值化”——许多像素为 0（近处），其余像素统一为配置的远裁面（`max_depth`，示例为 4.0）。训练路径未生成与 `camera_demo.py` 相同的可视化 / 保存输出。

采取的更改（回顾）：
- 在 `train.py` 中新增 `camera_monitor` 后台线程：周期性捕获并保存深度（wrapper 输出）、RGB、CSV、并写入详尽诊断统计（min/max/percentile/unique/counts/中心样本）。
- 为便于诊断，新增保存 renderer 原始 depth buffer 的逻辑：`camera_outputs/depth_raw_renderer_latest.npy`。这样可以判断二值化是否发生在渲染器层（renderer 原始就二值）或在 wrapper/后处理层。
- 允许通过 `cfg.camera` 控制 `CameraProperties` 的 `use_collision_geometry` 与 `enable_tensors`，可快速切换渲染几何（visual vs collision）与 CPU/GPU 路径做对照实验。
- 添加 `camera_test_mode`（短测模式）：在创建 env 前可临时覆盖 camera 参数并自动采集 N 帧然后提前退出，便于快速生成诊断输出而不跑完整训练。

关键诊断观察（示例）
- `depth_stats_latest.txt`（示例）显示：19200 像素中 16091 为 0.0，3109 为 4.0（max_depth=4.0），中心 5x5 区域全部为 0.0。说明中心指向近处几何，其余视野多为“未命中”导致被替换为 far_plane。

建议的后续动作
1. 在可导入 IsaacGym 的本地环境运行短测并上交 `camera_outputs/depth_raw_renderer_latest.npy` 与 `depth_stats_latest.txt`，以便判断 raw depth 是归一化深度（需反算到米）还是 renderer 本身就返回二值。
2. 若 raw 为归一化缓冲（0..1），在 wrapper 中加入从 depth-buffer 到线性米的转换（使用相机 near/far）；若 raw 已为米但仍二值，则通过参数扫参（相机位置/俯仰、`use_collision_geometry` 开关）定位原因。
3. 长期改进方向：添加异步抓帧与磁盘写入以支持大规模 env 下的稳定抓取；提供可选的 raycast-based depth 采样作为 renderer-independent 的后备。

此文件作为调试记忆，用于在未来快速回顾为何加入 `camera_test_mode`，为何保存 raw renderer depth，以及哪些配置项影响深度输出。
