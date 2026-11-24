# v_compression 开发与维护记录

更新：2026-11-24。当前版本：1.0.3。

## 项目范围

本项目用于模型基线评估、量化、剪枝与知识蒸馏实验。主要维护内容：压缩阶段调度、检测蒸馏、通用学生检测头适配、处理模式、模型验收与检查权重缓存。

业务后端、图形界面、HTTP 服务、任务日志、配置管理和运行协议在本项目内协同维护。模型框架和 adapter/Fork 属于声明的外部依赖。

## 维护入口

- 业务实现：`model_compression/`。
- 图形界面实现见[界面源码目录](../vtools_ui/)；React 源码和静态产物位于其中的 `webapp/frontend/`。
- 请求/结果协议实现见[运行协议目录](../vtools_runtime/)；接口文档见 [interface.md](interface.md)。
- 环境检查：`env_test/`；启动调度：`launcher.py`。
- 原生启动文件构建：`python packaging/build_launcher.py`。
- 前端修改后，在[前端工程目录](../vtools_ui/webapp/frontend/)执行 `npm ci` 和 `npm run build`。
- 可选目录包构建：`python packaging/build_app.py`。

发布名、窗口标题和启动文件使用 `v_compression`。

## 变更与验证

累计更新见 [VERSION.md](../VERSION.md)，当前功能边界和历史验证入口见 [function-scope.md](function-scope.md)。保存验证记录时写明日期、软件版本、运行设备与执行范围，不把历史结果改写为当前实测。

2026-11-20 增加 YOLO 检测 ONNX FP32、FP16、静态 INT8，以及同一验证集的精度门限和 CPU 对照。部署回归入口记录图简化、分段耗时、阈值和 CUDA 测试。象棋 FPS 实测中，原 YOLO11s 从 832 切换到 640 后由 32.0 提升到 33.6 FPS，验证集略多漏检，去重留出集略少漏检。等预算小模型训练、自适应裁图、结构剪枝及 K230 主机模拟结果见[验证记录](../verification/chess-deployment-2026-11-20.json)。

## 代码与依赖记录

实现与文档维护过程曾使用 AI 辅助；本记录不替代逐文件的作者与来源核对。许可证、已有署名和第三方通知随代码保留，依赖和模型权重的授权分别以实际采用版本为准。说明见 [第三方依赖与许可](../THIRD_PARTY_NOTICES.md)。
