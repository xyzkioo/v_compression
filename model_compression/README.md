# 模型压缩与知识蒸馏

## 项目入口

从本项目根目录运行以下命令。包入口使用版本 1 请求与结果协议；业务脚本提供原生参数接口。

```bash
python -m model_compression --help
python -m model_compression --list-modules
python -m model_compression --config /path/to/config.yaml --run-dir /path/to/new-job
```

`--request /path/to/request.json` 也可直接接收 JSON 请求，标准输出为统一结果，日志保存在新建的运行目录。
安装、参数表、结果 Schema 和可执行示例见 [接口说明](../docs/interface.md)。各模块的参数和输出见下文。

## 入口

从本项目根目录运行主入口，默认读取 `model_compression/config/config.yaml`（路径由程序定位，不依赖启动目录）：

```bash
python model_compression/run_model_compression.py
python model_compression/run_model_compression.py --list-modules
python -m model_compression --config model_compression/config/config.yaml --run-dir user-data/runs/new-job
```

每次运行都是独立实验：先做基线，再按模块顺序执行量化、剪枝或蒸馏。运行目录保存完整配置、汇总结果和全部模型产物；跨运行继续处理时，把上一次运行的产物路径填入 `model.weights`。本项目负责产生优化后的模型与评估结果；正式部署速度需在目标硬件上另行评测。

YOLO 检测模型的知识蒸馏使用 Ultralytics 原生 `model.train(..., distill_model=teacher.pt)`；本工具的 `distillation.classification` 只处理分类 logits。检测压缩的基线和产物精度通过 Ultralytics `YOLO.val()` 评估，结构化剪枝前后的固定输入计时仅用于同一次实验内比较，正式部署测速仍使用目标硬件上的测速工具。

## 快速开始

先安装 YAML 依赖和与本机匹配的 PyTorch。然后复制并修改 [config/config.yaml](config/config.yaml) 中的 `model.weights`，在本项目根目录运行：

```bash
python model_compression/run_model_compression.py
```

显式指定 `--device cuda:0` 等 CUDA 设备时，若设备不可用会报错；`auto` 才会在无 CUDA 时选择 CPU。`--run-dir` 需要指向尚不存在的目录，避免覆盖旧实验结果。

默认会在 `user-data/runs/runN/` 生成 `effective_config.json` 和扁平化的 `summary.json`。模型文件统一位于 `artifacts/structured_pruning`、`artifacts/unstructured_pruning`、`artifacts/distillation`、`artifacts/quantization` 或 `artifacts/multi_processing`。分类流程没有配置数据集时，基线仍会记录参数量并将评测标记为跳过；YOLO 检测基线需要填写 `dataset.data` 才能进行 mAP 评估。执行内容只由 `modules` 开关决定。

`effective_config.json` 的 `modules` 记录 YAML 与 `--only`、`--enable`、`--disable` 合并后的实际开关。结构化检测压缩启动前检查 CUDA 显存；总量低于 8 GiB 或当前空闲低于 6 GiB 时提示风险。日志标注剪枝前后评估、测速、依赖图追踪、微调和保存阶段，附阶段边界的 CUDA 空闲、已分配、已保留及本进程阶段峰值；OOM 错误会标明阶段。这不是精确容量预测，实际峰值还受输入尺寸、batch size、其他进程及模型结构影响。

典型运行目录如下：

```text
runN/
├── effective_config.json
├── summary.json
├── baseline_validation/              # 按需生成的评估图表
└── artifacts/
    ├── structured_pruning/
    ├── unstructured_pruning/
    ├── distillation/
    ├── quantization/
    └── multi_processing/
```

`summary.json` 只在 `model` 段记录一次公共模型信息，阶段结果使用 `baseline`、`unstructured_pruning`、`structured_pruning`、`distillation`、`quantization` 和 `multi_processing` 六个类别；每类按需包含 `parameters`、`metrics` 和相对于 runN 的 `artifact` 路径。

默认配置的任务是 YOLO 检测，因此动态 INT8 分类量化默认关闭；只有将 `model.task` 改为 `classify` 并准备分类数据后，才应显式启用 `compression.quantize.dynamic_int8`。检测压缩使用 `compression.prune.unstructured` 或 `compression.prune.structured`。

需要执行某个模块时，可以直接使用统一模块 ID：

```bash
python model_compression/run_model_compression.py --only compression.quantize.dynamic_int8
python model_compression/run_model_compression.py --only compression.prune.unstructured
python model_compression/run_model_compression.py --only compression.prune.structured
python model_compression/run_model_compression.py --only distillation.classification
python model_compression/run_model_compression.py --list-modules
```

显式传入 `--only ''` 表示关闭全部模块，不恢复 YAML 默认选择；不能与 `--enable/--disable` 混用。

也可以通过本项目的 JSON 协议入口调用：

```bash
python -m model_compression --config model_compression/config/config.yaml --run-dir user-data/runs/new-job
```

## 配置要点

`model.adapter: torch` 支持完整的 `torch.nn.Module` checkpoint，或在配置了 `model.factory: package.module:function` 时加载 `state_dict`。自定义网络和数据集可以复制 [adapters/torch_adapter_template.py](adapters/torch_adapter_template.py)，实现 `build_model`、`load_weights` 和可选的 `build_dataloader`。

自定义检测模型使用 [检测 adapter 模板](adapters/detection_adapter_template.py)。通用检测 adapter 可用非结构化剪枝、幅值通道剪枝、LAMP、Taylor 和 Fisher。支持原生 YOLO 接口的 adapter 可通过 `load_detector` 进入 YOLO 路径。具体函数和返回值见[检测 adapter 接口](../docs/interface.md#自定义检测-adapter)。

分类流程的 `dataset.train` 与 `dataset.val` 默认按 ImageFolder 目录读取；adapter 返回的 DataLoader 也可以产生 `(images, labels)` 或 `{"image": ..., "label": ...}`。YOLO 检测流程使用 `model.task: detect` 和 `dataset.data` 指向 `data.yaml`，保留每张图的多框标签。结构化方法见下表；依赖图方法物理删通道并同步调整关联层，检测头输出默认保留。`scale` 从模型 YAML 的 `scales` 按 `target_scale` 重建网络，建议提供对应规模的 `initial_weights`。默认精度门禁要求 mAP50–95 下降不超过 0.05，未通过的产物不能作为后续阶段输入。普通结构化方法可用 `quality_gate: false` 仅保留实验结果；速度搜索仍要求精度和延迟达标。微调和蒸馏若触发 AMP 检查，使用缓存 `~/.cache/ultralytics/weights/`（可用 `XDG_CACHE_HOME` 更改缓存根目录）。参数与依赖见 [config/config.yaml](config/config.yaml) 和 [REQUIREMENTS.md](REQUIREMENTS.md)。分类蒸馏使用：
```text
Loss = (1 - alpha) * CE + alpha * T^2 * KL
```

教师模型始终处于 `eval()` 且冻结参数，验证集只用于选择最佳 checkpoint。蒸馏结果记录教师和学生初始化权重路径。

配置键使用完整路径：结构化质量门禁字段为 `compression.structured.quality_gate`。

`summary.json` 按 `baseline`、`unstructured_pruning`、`structured_pruning`、`distillation`、`quantization` 和 `multi_processing` 分类保存参数、指标和相对产物路径。公共模型信息只在 `model` 中出现，完整配置只在 `effective_config.json` 中保留。多模块串行运行会额外保存最终模型到 `artifacts/multi_processing`；单模块结果仍保留在对应类别目录。

### 结构化剪枝方法

随附模板使用 `taylor` 作为结构化方法（模块开关保持原样）。已有配置的 `torch_pruning`、`scale` 保持可用。界面选“结构化剪枝”后可临时选择方法，也可通过 `--structured-method` 覆盖。

| 方法 | 控制参数与结果 |
| --- | --- |
| `taylor` | 在带标签训练集抽取 `calibration_images` 张，用检测损失梯度为依赖组评分。 |
| `fisher` | 用逐样本梯度平方的均方根替换 Taylor 中的有符号梯度，是近似评分而非完整 Fisher–Rao。 |
| `lamp` | 在候选层间分配 `channel_sparsity` × 候选层输出通道总数，单层上限由 `lamp_stage_cap` 控制；不等于整网参数比例。 |
| `bn_gamma` | 实验项：用 `bn_sparse_*` 参数先做分层 L1 稀疏训练，再按 BN 缩放系数绝对值物理剪枝，可选剪后微调。 |
| `latency_search` | 在 `search_candidates` 上限内枚举通道和可跳过的残差块候选，用验证集 mAP50–95、同设备 PyTorch 交错测速与 P95 共同过门禁。候选和复测报告在 `artifacts/structured_pruning`；没有达标结构就保留原模型，不推荐剪枝产物。 |

`protected_stages` 排除指定的顶层索引，不将其作为通道删除根或残差块跳过的候选；依赖图仍可能调整这些层的输入。用 `finetune_epochs: 0` 可跳过剪后训练。训练相关方法使用标准 YOLO images/labels 目录和带标签的 train/val；校准数据不会取自验证集。

跨运行不再自动选择父模型或维护分支指针。需要继续处理时，将上一运行 `summary.json` 中的相对 `artifact` 路径与该 runN 目录拼成实际路径，再作为下一运行的 `model.weights`。

非结构化剪枝只改变权重稀疏度，不自动声称文件变小或推理变快；需要在目标硬件上用 `speed_test` 实测。

## 检测知识蒸馏

设置 `model.task: detect`，填写 `distillation.teacher_weights`、`distillation.student_weights` 和 `dataset.data`，启用 `distillation.detection`。教师与数据集的类别数量、名称和顺序必须一致；学生可使用通用预训练模型，程序会重建目标类别输出并迁移兼容权重，再检查蒸馏特征层兼容性。

本次任务保存 `student-initialized.pt`，用于学生基线和训练；原始权重不覆盖。训练后重新加载学生产物并核对结构，报告记录前后指标和实际输入来源。选择该模式不需要额外公共模型权重。训练配置详见 [配置文件](config/config.yaml)，操作步骤见 [操作说明](../docs/user-guide.md)。
