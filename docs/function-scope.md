# v_compression 1.0 功能与验证范围

更新日期：2026-11-03。

## 软件功能

组织模型压缩实验，执行基线评估、量化、剪枝或知识蒸馏并保存产物。

输入与操作：选择 detect 或 classify，填写权重与匹配的数据集。检测实验使用检测数据集 YAML；分类实验使用按类别分目录的训练/验证集。只启用本次需要的模块。检测蒸馏还需支持该功能的 Fork、教师权重与学生初始化权重。

输出：effective_config.json、summary.json，以及 artifacts 下的量化、剪枝或蒸馏模型产物；评估图表按配置与数据集生成。

本项目的业务实现包括：基线与阶段调度、配置校验、量化层统计、结构缩放和依赖感知剪枝接入、蒸馏训练流程、指标汇总、产物管理及来源记录。

本项目提供图形界面、HTTP 服务、任务日志和请求/结果协议。模型框架与相关依赖见 [依赖说明](../THIRD_PARTY_NOTICES.md)。

## 实现边界

分类动态 INT8 主要处理 Linear、LSTM 和 GRU 等层。检测模型可导出 FP32、FP16 和静态 INT8 ONNX，报告同一验证集上的 mAP、体积和 ONNX Runtime CPU 延迟。检测蒸馏使用支持原生检测蒸馏的 Ultralytics Fork。

## 已有验证证据

已验证小分类模型 CPU 动态 INT8 与模型保存。0.1.5 另用实际的 16 类象棋教师与通用 80 类 YOLO26n，在合成数据上完成一轮 CPU 原生检测蒸馏，确认学生自动适配、最终独立模型加载及输入哈希不变；错误的教师/数据集搭配会在训练前拒绝。这些检查不代表完整数据集精度或 GPU 性能已验证。记录见 [detection-distillation-0.1.5.json](../verification/detection-distillation-0.1.5.json)。

项目目录、HTTP/API、配置编辑、CPU 任务、日志和失败状态的历史检查记录见 [independence.json](../verification/independence.json)。原生窗口在 0.1.3 的验证见 [native-window.json](../verification/native-window.json)；0.1.4 回归见 [repair-0.1.4.json](../verification/repair-0.1.4.json) 与 [source-repair-0.1.4.json](../verification/source-repair-0.1.4.json)。历史记录保留其实际版本，不改写为当前版本。

任何“实现了”的描述均不自动表示在所有模型、设备和数据集上实测成功。本文为软件技术说明，包含 AI 协助修订，不是未使用 AI 的申报承诺。
