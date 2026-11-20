# v_compression 1.0 操作说明

## 用途与准备

组织模型压缩实验，执行基线评估、量化、剪枝或知识蒸馏并保存产物。

程序面向 Linux x86_64 文件夹运行。整个文件夹保留在可写位置，双击 `v_compression` 启动。默认已有前端产物，无需 Node。界面解释器需安装根目录 `requirements.txt`；模型解释器还需对应业务依赖。配置样例位于 `model_compression/config/config.yaml`，输入放 `inputs/` 或选择外部文件。

## 图形操作

1. 打开“设置”，确认“模型任务使用的 Python 解释器”和“默认结果目录”，点击“保存设置”。默认运行数据在本项目 `user-data/`。
2. 打开“模型压缩”。选择 detect 或 classify，填写权重与匹配的数据集。检测实验使用检测数据集 YAML；分类实验使用按类别分目录的训练/验证集。选择本次处理模式，一次只运行一种处理方式。检测蒸馏还需支持该功能的 Fork、教师权重与学生初始化权重。
3. 通过“图形化配置”编辑配置；保存后选择模式并启动任务。若尚未填写权重或数据路径，应先补齐配置，默认模板不会自行下载模型。
4. 在底部“运行详情”查看日志。运行失败时检查日志中的路径、设备和依赖错误；正在运行的任务可点击“停止任务”。
5. 查看完成状态及结果目录，使用结果查看页面浏览文件；历史任务可查看保存的日志。任务完成不等于模型指标达标。

图形界面按模式显示输入：基线评估、结构化剪枝、非结构化剪枝使用公共权重；检测蒸馏只显示教师、学生和数据集，不需要重复填写公共权重。模式会自动安排所需评估及保存步骤。YAML 的模块开关不会与所选模式叠加；同一份 YAML 中可以保留其他算法参数。

目标检测默认打开“结构化剪枝”的方法列表。可单独勾选方法，也可点“全选”勾选全部七种方法。“跟随配置文件”单独使用，执行 YAML 里配置的方法。勾选多种时，各方法用同一份输入权重依次独立运行，输出按方法分目录保存。鼠标停在问号上可查看方法原理、所需数据和关键参数，移开即隐藏。键盘可用 Tab 查看、Esc 关闭。

Taylor、Fisher、BN-γ 和速度搜索需要有标签的训练集；试剪时建议先把剪后微调轮数设为 0。BN-γ 即使跳过剪后微调也会先进行稀疏训练。速度搜索以本机 PyTorch 固定输入的纯前向延迟为准，未通过精度或延迟复测时保留候选报告并提示保留原模型。

自定义检测模型通过 YAML 的 `model.adapter` 选择 adapter。工具自动执行参数统计和基线评估，并用同一个 adapter 评估保存后重新加载的剪枝模型。通用 PyTorch 检测 adapter 可以使用非结构化剪枝、幅值通道剪枝、LAMP、Taylor 和 Fisher。接入条件和模板见[检测 adapter 接口](interface.md#自定义检测-adapter)。

检测模型可选「模型量化」，填写 YOLO `.pt` 和检测数据集 YAML，然后选择 FP32、FP16 或静态 INT8。FP32 导出基线，FP16 生成半精度图，INT8 从 `train` 取样校准卷积。模型和 `onnx-comparison.json` 位于业务运行目录的 `artifacts/quantization/`，可对照两份 ONNX 的 mAP、文件体积和 CPU 前向延迟。FP16 图在 CPU 上的算子执行精度和速度由运行时决定。

## 部署回归测试

在源码仓库根目录，用同一批真实图片检查固定尺寸 ONNX。报告记录读图解码、预处理、前向、后处理和整段延迟的 P50、P95，以及端到端 FPS。FPS 使用预热后帧数除以总计时，同一轮中的 ONNX 输入尺寸必须匹配。指定 `--labels` 后另记录各组置信度和 NMS 阈值下的误检、漏检与召回率。输入按 YOLO 标注约定配对，阈值对照使用验证集。

```bash
python -m model_compression.tests.deployment_probe --reference /path/model-fp32.onnx --variant fp16=/path/model-fp16.onnx --variant int8=/path/model-int8.onnx --images /data/images/val --labels /data/labels/val --run-dir user-data/runs/deploy-probe --simplify --cuda-nms --cuda-preprocess --cuda-weights /path/best.pt
```

`--simplify` 需要安装 `onnxslim`，执行图简化、张量与检测结果一致性检查。CUDA 项需要可用的 PyTorch CUDA，GPU NMS 还需要配套的 `torchvision`。报告分别记录显存张量做 NMS、已解码图片的上传和预处理，以及双缓冲上传与推理。整段计时覆盖图片文件到检测框，保存在新目录的 `deployment-comparison.json`。板端验收使用目标设备及其运行时重复采集耗时和精度。

比较 PyTorch 输入尺寸时，可固定同一个权重和 batch=1，用 Ultralytics 分别设置 `imgsz=832`、`640` 和 `416`。先预热每个尺寸，在同一批已解码图片上交替执行各方案，再记录同步后的 FPS 和 P95。使用相同的 `rect=False`、`conf=0.25` 和 `iou=0.70` 对照漏检。象棋实验的计时范围、精度和结果见[验证记录](../verification/chess-deployment-2026-11-20.json)。

## 命令行与文件接口

检测蒸馏可直接选择通用预训练学生（如 80 类的 `yolo26n.pt`），无需先训练小模型。教师与数据集的类别名称和顺序必须一致；程序自动把学生检测头适配为该数据集的类别，并在本次运行保存 `student-initialized.pt`，不覆盖原始权重。适配后的初始化用于学生基线和蒸馏训练；类别不匹配的数据集会明确报错。仍需选用与教师蒸馏特征层兼容的学生架构。

在本项目根目录执行：

```bash
python -m model_compression --help
python -m model_compression --config model_compression/config/config.yaml --run-dir user-data/runs/example-new-run
python -m model_compression --request /path/to/request.json
```

`example-new-run` 必须尚不存在；每次实验使用新目录。接口字段、相对路径和退出码见 [interface.md](interface.md)。控制台可用 `./启动.sh --check` 确认导入位置。

## 结果与限制

图形任务与上面的 `python -m` 入口均使用统一结果目录。先看 `result.json`，按其中 `artifacts[].path` 相对本次运行目录定位业务文件；下面列出的报告位于 `artifacts/` 内，可能还有业务子目录。原生脚本的目录层级见业务 README。

effective_config.json、summary.json，以及 artifacts 下的量化、剪枝或蒸馏模型产物；评估图表按配置与数据集生成。

分类动态 INT8 主要处理 Linear、LSTM 和 GRU 等层。检测静态 INT8 报告记录有 INT8 权重的卷积数和本机推理延迟。检测蒸馏使用支持原生检测蒸馏的 Ultralytics Fork。

已验证小分类模型 CPU 动态 INT8 与模型保存，以及通用学生自动适配后的一轮合成数据 CPU 检测蒸馏。完整数据集精度、大模型剪枝与 GPU 性能仍需按实际配置验证。更完整的范围见 [function-scope.md](function-scope.md)。

## 故障排查

双击启动失败时查看 `user-data/launcher.log`；也可在终端执行 `./启动.sh` 观察错误。文件选择按钮需要桌面窗口桥接；浏览器模式可直接填写绝对路径。更换模型 Python 后保存设置，并重新运行环境检查。移动项目时移动整个文件夹，外部权重、数据或解释器的绝对路径需要重新选择。
