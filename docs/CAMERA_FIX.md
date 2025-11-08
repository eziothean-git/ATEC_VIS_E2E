Camera notes and fixes

- 新增配置项 `cfg.camera.print_perf`，默认值为 `False`。当你需要输出每 N 步的性能汇总（带有 `[Perf]` 前缀）用于调试时，将其设为 `True`。
- 该配置仅影响周期性打印，不会改变运行时逻辑或性能路径。
