# 视觉模型轻量化工具：技术调研与首版选型

调研日期：2026-11-19。范围：现有 PyTorch 视觉模型，优先支持 YOLO 目标检测，图像分类用于低成本回归。本文是选型与验证计划，未安装新依赖、训练模型或实测速度。

## 1. 定位：做部署工作流，不重写压缩库

给定模型、目标设备和允许的精度损失，生成并比较可运行的部署产物。首版先打通 **FP32 ONNX 基线、ONNX Runtime CPU 静态 INT8、结构化通道剪枝**。已有蒸馏作为可选的精度恢复手段；若首要设备是 NVIDIA GPU，则将 TensorRT FP16 提到首批验证，INT8 后接。以目标机实测为准，不把参数量和 MACs 当成加速承诺。参见 [NetAdapt，ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/html/Tien-Ju_Yang_NetAdapt_Platform-Aware_Neural_ECCV_2018_paper.php)。

| 现有能力 | 新工具中的用法 |
| --- | --- |
| `compression.quantize.dynamic_int8` | 保留作 CPU 对照。默认量化 `Linear`，可配置 RNN 类层，不是 YOLO 卷积的静态量化后端。 |
| `compression.prune.unstructured` | 保留作稀疏度实验；普通稠密产物不承诺文件缩小或推理加速。 |
| `compression.prune.structured` | 复用依赖图剪枝、微调与精度门禁。`scale` 按配置重建小模型，`torch_pruning` 物理删除通道，两者在报告中分开。 |
| `distillation.classification` / `distillation.detection` | 复用训练流程；检测蒸馏以实际接入的 Ultralytics 版本或 Fork 为准。 |
| 请求/结果协议、基线与运行目录 | 通过版本化接口接入，输入只读，候选各有独立产物和日志，不跨项目修改 `sys.path`。 |

已有边界见[模块说明](../model_compression/README.md)、[功能与验证范围](function-scope.md)和[接口说明](interface.md)。新工具的核心增量是导出一致性、代表性校准、精度门禁、目标硬件测速和候选对比。

## 2. 方法、论文与实现

独立剪枝框架、YOLO 专用实现、延迟约束论文和兼容性核对见[视觉模型剪枝专项调研](pruning-implementations.md)。

| 方法 | 代表论文 | 实现入口与建议 |
| --- | --- | --- |
| 结构化通道剪枝 | [DepGraph，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html) | **首版。** 复用 [Torch-Pruning](https://github.com/VainF/Torch-Pruning/blob/master/README.md)；参考 [YOLOv8 剪枝示例](https://github.com/VainF/Torch-Pruning/blob/master/examples/yolov8/yolov8_pruning.py)，针对实际 YOLO 版本验证微调、重载和导出。 |
| 通道重要性与剪枝率选择 | [Network Slimming，ICCV 2017](https://arxiv.org/abs/1708.06519)；[Taylor Pruning，CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Molchanov_Importance_Estimation_for_Neural_Network_Pruning_CVPR_2019_paper.html)；[Isomorphic Pruning，ECCV 2024](https://arxiv.org/abs/2407.04616) | **进阶。** 先用组权重范数，再试 BN 缩放因子稀疏训练、梯度敏感度或同类结构组内排名。参考 [Slimming](https://github.com/Eric-mingjie/network-slimming)、[Taylor](https://github.com/NVlabs/Taylor_pruning)、[Isomorphic](https://github.com/vainf/isomorphic-pruning)；研究代码先核对许可。 |
| 非结构化剪枝与存储编码 | [Deep Compression，ICLR 2016](https://arxiv.org/abs/1510.00149) | **保留对照。** [PyTorch pruning](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html?highlight=prune%2F1000) 提供幅值置零；`prune.remove()` 不会缩小张量维度。论文中的存储编码也不等于推理时自动加速。 |
| 半结构化稀疏：2:4 | [NVIDIA 稀疏 INT8 训练与部署说明](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration) | **硬件插件。** 参考 [PyTorch 2:4 示例](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html) 和 [TensorRT 稀疏日志](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/inference-library/data-formats-tensors.html)。除了检查模式，还要确认引擎选用的内核和实测收益。 |
| 静态 INT8 PTQ（训练后量化） | [Jacob 等，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)；[Nagel 等量化综述，2021](https://arxiv.org/abs/2103.13630) | **首版。** [ONNX Runtime `quantize_static`](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html) 输出 Q/DQ（量化/反量化）图；Intel 设备可对照 [OpenVINO + NNCF YOLO 示例](https://docs.openvino.ai/2024/notebooks/yolov8-object-detection-with-output.html)。 |
| FP16 与混合精度 | [ONNX Runtime FP16/混合精度](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) | **NVIDIA 首批对照。** 不需要 INT8 校准，可保留敏感节点的高精度。FP16 是浮点降精度；以 FP16 保存权重也不代表所有算子以 FP16 执行。 |
| 进阶 PTQ 与 QAT（量化感知训练） | [AdaRound，ICML 2020](https://proceedings.mlr.press/v119/nagel20a)；[BRECQ，ICLR 2021](https://arxiv.org/abs/2102.05426)；[Jacob 等，CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html) | **第二阶段。** 先试校准和敏感节点回退，再研究 AdaRound/BRECQ 重构，或用 QAT 微调。参考 [BRECQ 实现](https://github.com/yhhhli/BRECQ/blob/main/quant/block_recon.py)、[TorchAO PT2E](https://docs.pytorch.org/ao/stable/pt2e_quantization/index.html)、[NVIDIA ModelOpt](https://github.com/NVIDIA/Model-Optimizer)；先选部署后端再定 QAT 导出路径。 |
| 知识蒸馏 | [Hinton 等，2015](https://arxiv.org/abs/1503.02531)；[DKD，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Decoupled_Knowledge_Distillation_CVPR_2022_paper.html)；[CWD，ICCV 2021](https://arxiv.org/abs/2011.13256) | **复用现有流程。** 参考 [PyTorch 蒸馏示例](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html)和 [Ultralytics 检测蒸馏](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/knowledge-distillation.md)。轻量来自学生结构，蒸馏主要用于减少精度损失。 |
| 低秩分解 | [Jaderberg 等，2014](https://arxiv.org/abs/1405.3866)；[Lebedev 等，2015](https://arxiv.org/abs/1412.6553) | **后续实验。** `Linear` 可用 [SVD](https://docs.pytorch.org/docs/stable/generated/torch.linalg.svd.html) 截断，卷积可试空间分解或 CP。会引入中间张量和额外算子，先定位热点再验收。 |
| 轻量架构与结构重参数化 | [MobileNetV3，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)；[RepVGG，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html) | **小模型作为对照。** [TorchVision MobileNetV3](https://docs.pytorch.org/vision/master/models/mobilenetv3.html) 可做分类回归；[RepVGG `switch_to_deploy`](https://github.com/DingXiaoH/RepVGG) 只转换兼容块。检测比较原生小版 YOLO 与剪枝模型。 |
| 部署图优化 | [ONNX Runtime 图优化](https://onnxruntime.ai/docs/performance/graph-optimizations) | **首版基础设施。** 优先用后端常量折叠与融合；核对导出前后结果，不把未经单独测量的融合收益归给剪枝或量化。 |

## 3. 两条可落地的路线

### 路线 A：YOLO → FP32 ONNX → 静态 INT8

1. 从可信 `.pt` 导出 FP32 ONNX，或读入具有明确输入/输出契约的 `.onnx`。首版限定已验证的固定输入尺寸与 batch；锁定预处理（颜色、归一化、letterbox）和后处理（解码、NMS 的位置与参数，若模型使用 NMS）。量化前，先对齐 PyTorch 与 FP32 ONNX 的输出和整集指标。
2. 用 `CalibrationDataReader` 送入与部署预处理一致的代表性图像；普通 PTQ 校准图像不需要标签。记录样本清单和预处理；校准与调参验证分开，最终结果用未参与调参的测试集。
3. `quantize_static()` 先试 Q/DQ + MinMax，再按模型和 CPU 比较逐通道、符号组合、Entropy 或 Percentile。确认节点敏感后，用 `nodes_to_exclude` 保留原精度；也可参考 [NNCF `quantize_with_accuracy_control`](https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training/quantizing-with-accuracy-control.html)。见 [ORT 量化 API 源码](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py)。
4. 重新加载产物、跑整集验证和[后端 profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)。分别记录 Q/DQ 图覆盖、能确认的实际执行精度和整模延迟，不以节点个数宣称全模型 INT8。
5. NVIDIA 独立验证：先比 TensorRT FP16，再用 [TensorRT 显式量化](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html)配合 ModelOpt，或使用经验证的 [Ultralytics 导出](https://github.com/ultralytics/ultralytics/blob/main/docs/en/integrations/tensorrt.md)。保留 ONNX，按目标 TensorRT/CUDA/GPU 与形状范围构建 engine，不能默认复用 ORT CPU 配方或其他设备生成的 engine。见 [TensorRT 兼容矩阵](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)。

### 路线 B：依赖图剪枝 → 微调 → 导出

1. 复用依赖图路径，用代表性输入形状追踪连接，保护检测头输出及类别维度。只修改通过检查的通道组；遇到不支持的自定义层、分支或切分时明确报错，不默默退化为置零。
2. 试小幅度候选（例如目标通道比例 10%、20%、30%，记录实际删减量），与原始和原生小版 YOLO 对照。保留通道变更清单、结构描述与重载方法；通过结构检查后微调，必要时再开蒸馏。
3. 重载剪枝模型、验证并导出 FP32 ONNX，与未剪枝模型在同一后端测速。若叠加 INT8，在剪枝与微调后重新校准；分别保留「剪枝」「INT8」「剪枝 + INT8」结果，区分各阶段收益。

## 4. 首轮实验与验收

| 阶段 | 可验证交付 |
| --- | --- |
| 0：导出和测量基线 | 选一个已有 YOLO 权重和真实数据集，打通 PyTorch FP32 / ORT FP32 输出对齐、同口径验证和 CPU 测速；小型分类模型用于自动化回归。 |
| 1：可部署轻量化 | 增加 ORT CPU 静态 INT8 和依赖图剪枝；重载产物、评估精度和测速。校准数据缺失、运行失败、不可测或精度超差均有明确状态。 |
| 2：扩展设备与精度恢复 | 根据设备选 OpenVINO/NNCF 或 TensorRT，不强制安装所有后端；按瓶颈接入敏感层回退、蒸馏或 QAT。 |

- **候选：** 原始 `.pt` FP32、同模型 ONNX FP32、ONNX INT8、剪枝后 ONNX FP32、剪枝后重校准的 ONNX INT8、原生小版 YOLO ONNX FP32。对照原生小模型时用可比的数据、输入尺寸和后训练条件，并记录差异。
- **精度：** 检测至少记录 `mAP50–95`、`mAP50`，需要时补每类 AP；分类至少 Top-1。门禁记绝对百分点，例如 `0.420 → 0.410` 是下降 **1.0 个百分点**，不是相对下降 1%。阈值由任务定义，超差候选可保留记录但不标为已验收。
- **性能：** 固定设备、后端、版本、batch、尺寸、线程数和预热次数，报告 P50/P95、吞吐、可测时的峰值内存/显存。纯推理与含预/后处理的端到端延迟分开；GPU 计时使用同步边界。同后端对比优化净收益，跨后端结果单列。
- **产物：** 记录实际总字节（包括外部权重、IR 文件等）、参数量与 MACs 的统计口径、输入哈希、依赖版本、数据拆分、配置及失败日志。复用运行目录与 `summary.json`，汇总精度、体积、延迟；没有满足门禁就报告未达标。

**建议阅读顺序：** DepGraph → ORT 量化文档和 OpenVINO YOLO 示例 → Jacob 量化论文 → NetAdapt → 按精度瓶颈查蒸馏、AdaRound/BRECQ 或 QAT。移动端与 LLM 不列入首版；可从 [LiteRT 量化](https://developers.google.com/edge/litert/quantization/model_optimization)和 [AWQ](https://hanlab.mit.edu/projects/awq)另立调研分支。
