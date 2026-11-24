# 模型压缩 接口 v1（软件 1.0.2）

## 输入

`config` 与 `run_dir` 必填；`inputs` 和 `options` 可省略。下面是关闭全部业务模块的协议检查示例，不是模型测试示例。将其保存为本项目根目录的 `request.json`，再执行 `python -m model_compression --request request.json`。相同 `run_dir` 只能运行一次，重跑时请改用新目录。

```json
{
  "schema_version": 1,
  "component": "model_compression",
  "config": "model_compression/config/config.yaml",
  "run_dir": "user-data/runs/example-new-run",
  "inputs": {},
  "options": {
    "only": []
  }
}
```

`component` 固定为 `model_compression`。输入字段为 `weights`、`data`、`train`、`val`、`teacher_weights`、`student_weights` 和 `structured_initial_weights`。选项包括 `device`、`task`、`structured_scale`、`structured_method`、`structured_methods`、`structured_epochs`、`quantization_precision`，以及模块 ID 数组 `only`、`enable`、`disable`。`only: []` 表示关闭所有模块，`only` 与启用或关闭选项互斥。

`structured_method` 可为 `taylor`、`lamp`、`fisher`、`bn_gamma`、`latency_search`、`torch_pruning` 或 `scale`。它覆盖本次 YAML 的 `compression.structured.method`，不修改原 YAML。

`structured_methods` 接收非空且无重复的方法 ID 数组，与 `structured_method` 互斥。CLI 可用 `--structured-methods taylor lamp`。仅适用于目标检测的结构化剪枝模式。省略 `only` 时执行结构化剪枝、参数统计和基线评估，显式指定时必须包含结构化剪枝，也可包含参数统计和基线评估。多方法任务依次独立运行，各自使用同一输入权重。子目录为 `artifacts/<方法 ID>/`，批次汇总位于 `artifacts/structured-methods.json`。某个方法失败时继续执行其他方法，整批以退出码 2 报告失败并保留已生成的结果。

请求文件内的路径相对该 JSON 文件目录；直接 CLI 路径相对工作目录。原生 YAML 内输入路径相对 `project.root`（该字段相对 YAML 目录）。`run_dir` 必须是不存在的新目录，不能覆盖输入。开展实际业务时，删除 `only: []` 以沿用 YAML 的模块开关，或在 `only` 中填写需要的模块 ID；同时在 YAML 或 inputs 中填写对应模型/数据。空模块检查成功不证明这些模型、数据和业务功能可用。

## 图形处理模式

`/api/runs` 的表单 `values.processing_mode` 可选 `baseline`、`structured`、`unstructured`、`detection_distillation`、`detection_quantization`、`quantization` 或 `classification_distillation`，按模型任务限制可用项。处理模式被转换为现有协议的 `options.only`，不能同时附带 `module_edits`。

结构化剪枝模式的表单 `values.structured_methods` 传入勾选的方法 ID。单个方法沿用单次运行的输出结构。选择“跟随配置文件”时省略方法字段。

检测蒸馏模式只选择 `distillation.detection`，忽略隐藏的公共权重输入；剪枝和量化模式只选择对应的一种变换及参数统计、基线，不带隐藏的教师/学生输入。旧 CLI/JSON 请求仍支持显式模块组合；若只做检测蒸馏，冗余的公共参数统计与基线会报告 `skipped`，不再要求 `model.weights`。蒸馏内部的学生适配、参数统计和学生基线仍会执行。

## 自定义检测 adapter

在 YAML 中设置 `model.task: detect`，将 `model.adapter` 设为 Python 模块名或相对 `project.root` 的 `.py` 路径。权重仍用 `model.weights` 或界面的公共模型权重。可复用现有 adapter 的 PyTorch 模型构建与输入生成函数。示例见[通用检测 adapter 模板](../model_compression/adapters/detection_adapter_template.py)。

| 接口 | 什么时候用 |
| --- | --- |
| `build_model(weights, device)` 或 `load_model(weights, device)` | 加载原始和剪后的 checkpoint，返回 `torch.nn.Module`。还可沿用分类 adapter 的 `build_model` 和 `load_weights` 组合。 |
| `evaluate_detection(model, config, split, device, run_dir)` | 基线评估和剪后评估。返回 0 到 1 的 `map50_95`，可同时返回 `map50`、`precision` 和 `recall`。使用传入的模型和划分，评估产物写入 `run_dir`。数据集由 adapter 按配置读取。 |
| `pruning_targets(model, config)` | 结构化剪枝。返回可剪普通 `Conv2d` 的完整层名列表，避开检测头输出和耦合层。`protected_stages` 的索引对应这个列表。旧的单参数写法也可用。 |
| `make_inputs(batch_size, image_size, device, dtype)` | 可选。为追踪和测速返回 Tensor、位置参数序列、关键字参数字典或带 `args` 和 `kwargs` 的 InputBundle。默认输入是单张 NCHW RGB 张量。模型前向应有可追踪的可微输出。 |
| `calibrate_pruning(model, method, config, device)` | Taylor 和 Fisher 必需。根据有标注的训练样本将检测损失梯度写到参数的 `.grad`。Taylor 可累积批梯度，Fisher 使用逐样本梯度平方均值的平方根。 |
| `finetune_detection(model, epochs, config, device, run_dir)` | `finetune_epochs > 0` 时必需。返回保持剪后结构的模型。设为 0 可做快速试剪。 |
| `save_model(model, path)` | 可选。若提供，加载器需能恢复变化后的通道结构。默认保存完整 PyTorch 模型，加载时需使用原 adapter 和模型类。 |
| `forward_for_pruning(model, *args, **kwargs)` | 可选。返回供依赖图追踪的特征。追踪骨干时应返回供检测头使用的全部特征。测速和评估仍使用完整模型。 |

幅值通道剪枝、LAMP、Taylor 和 Fisher 复用依赖图删通道。工具检查固定测试输入下的输出形状，并在保存后检查重新加载的参数形状。分层 BN-γ、通道 + 残差块搜索和模型规格缩放依赖 YOLO 训练器或结构。使用这三种方法的自定义 adapter 可提供 `load_detector(weights, device, config)`，返回可供 Ultralytics 评估、训练、保存和重载的 YOLO 检测实例，且模型仍需具备对应方法要求的层、BN 或规格表。检测蒸馏使用原生 Ultralytics Fork。

### TorchVision 风格模型

选择 `model.adapter: model_compression.adapters.detection_adapter_template`。模型训练时接收 `images, targets` 并返回标量损失字典，推理时每张图返回含 `boxes`、`scores`、`labels` 的字典。边界框使用原图像素坐标，前景类别 ID 与 COCO 标注一致。模板使用 `pycocotools` 评估 bbox mAP，用 AdamW 在剪后结构上微调。

- `dataset.train` 和 `dataset.val` 指向 COCO 标注 JSON，`dataset.root` 指向图片目录。图片分开放时可用 `dataset.image_roots.train` 和 `dataset.image_roots.val`。
- `compression.structured.pruning_targets` 填入骨干内部可剪层全名，如 `[backbone.0]`。依赖图追踪骨干，保护供检测头消费的各输出形状。
- 输入可为完整 PyTorch checkpoint，或通过 `model.factory` 构建模型来加载 `state_dict`。剪枝产物保存完整结构，便于重载。

### 含自定义层的 YOLO checkpoint

选择 `model.adapter: model_compression.adapters.ultralytics_custom_adapter`，将自定义层源码目录放入 `project.python_paths`，在 `model.adapter_options.import_modules` 填写模块名。可用 `model.adapter_options.required_layers` 验证重载后的模型仍包含指定层。检测评估、校准和微调继续使用 YOLO 原生流程。规格缩放需要模型 YAML 能重建自定义结构。

## 检测 ONNX 精度

`compression.quantize.onnx` 接收 Ultralytics YOLO `.pt` 检测权重，也支持有 `load_detector` 的 YOLO 兼容 adapter。部署输入固定为 batch 1，方形尺寸由 `dataset.input_size` 指定且可被模型步长整除。YOLO 数据集 YAML 用于复评，INT8 从其 `train` 划分取样。

YAML 使用 `compression.quantization.precision`，默认 `int8`。可用请求的 `options.quantization_precision` 或 CLI 的 `--quantization-precision` 覆盖，取值为 `fp32`、`fp16` 或 `int8`。

```bash
python -m model_compression --config model_compression/config/config.yaml --run-dir user-data/runs/onnx-int8 --weights /path/to/best.pt --data /path/to/data.yaml --only model.parameters,baseline.evaluate,compression.quantize.onnx --quantization-precision int8
```

产物为 `model-fp32.onnx` 和所选精度的 ONNX。FP16 保持浮点输入和输出，INT8 使用 ORT 静态 Q/DQ、MinMax 校准和默认逐通道卷积权重量化。`calibration_images` 控制取样数，`nodes_to_exclude` 可保留敏感节点。输出复载后用 ONNX Runtime CPU 测速，并在同一验证集计算精度。若 `onnx-comparison.json` 的 mAP50–95 降幅超出 `max_map50_95_drop`，任务标记失败并保留诊断报告。

## 输出

检测蒸馏的 `student_weights` 可指向通用预训练小模型（例如 80 类的 YOLO26n），不要求先在目标数据集训练。程序先校验教师与 `dataset.data` 的类别数量、名称和顺序，再按数据集重建学生检测头并迁移兼容权重；新增类别的分类输出保持初始化，已匹配类别可迁移对应行。类别数相同但顺序不同的学生也会适配。仅改 `nc` 数字不构成适配。无法从 YAML 忠实重建的已剪枝主干会明确报错。

适配初始化保存在本次运行的 `artifacts/distillation/student-initialized.pt`（相对于原生后端运行目录），学生基线与训练使用同一检查点，不覆盖输入权重。`summary.json` 的检测蒸馏指标包含 `student_initialization`：`adapted`、`source_nc`、`target_nc`、`names`、`checkpoint`，适配时还含 `transferred_tensor_count`。最终网络结构与适配后的学生核对。教师与数据集不匹配时停止，不修改教师检测头。

stdout 只输出一个最终 JSON 对象；`--stream-logs` 将实时后端日志送 stderr。退出码 0：执行成功或显式跳过；2：执行失败；130：取消。此包入口当前不返回业务检查退出码 1。成功不等于模型指标达标，需检查领域报告。

运行目录包含 `request.json`、`config.snapshot.yaml`、`result.json`、`stdout.log`、`stderr.log` 和 `artifacts/`。结果包含 `schema_version`、`component`、`component_version`、`status`、`exit_code`、`backend_exit_code`、`error`、`run_dir`、`artifacts` 与时间。输入校验失败时不创建目录，最终 JSON 返回 `run_dir: null`。JSON Schema 见本项目的[请求定义](../vtools_runtime/schemas/request-v1.schema.json)和[结果定义](../vtools_runtime/schemas/result-v1.schema.json)。

业务文件位于 `artifacts/` 内；具体相对路径以 `result.json` 的 `artifacts[].path` 为准，将该路径拼接到本次 `run_dir` 即可定位文件。直接执行业务原生脚本时没有统一协议外层目录，其业务报告可以直接位于原生运行目录。

后端创建业务目录后异常退出时，组件会在缺少原生摘要的情况下补写 `artifacts/summary.json`，记录退出码及日志中最后运行的模块和阶段。
