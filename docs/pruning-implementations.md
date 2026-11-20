# 视觉模型剪枝：独立实现与 YOLO11 选型

核对日期：2026-11-19。范围：PyTorch/YOLO11 检测模型，优先找实际缩小张量的实现；也记录容易被误认为同类的掩码和硬件稀疏方案。下表是上游论文与源码核对，不是这些库在本地象棋模型上的运行结果。

## 独立框架与检测模型实现

| 实现 | 怎么决定剪什么、怎么产出模型 | 对本项目的适用性 |
| --- | --- | --- |
| [NVIDIA ModelOpt FastNAS](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.prune.pruning.html) | 将支持的网络转换成可搜索的结构，用评估函数为候选打分，在参数量、FLOPs 等约束下选子网并导出；[NAS 指南](https://nvidia.github.io/Model-Optimizer/guides/7_nas.html)提供示例。 | **优先做独立实现对照。** 公开 API 的约束列表不是实测延迟；不能直接视为 YOLO11 已兼容，也不能用分类示例替代检测 mAP 和导出测试。 |
| [OpenMMLab MMRazor Group Fisher](https://github.com/open-mmlab/mmrazor/blob/main/configs/pruning/base/group_fisher/README.md) | [Group Fisher，ICML 2021](https://arxiv.org/abs/2108.00708)：训练时累计耦合通道组的 Fisher 重要性，用激活内存或 FLOPs 变化做归一化，有剪枝、微调、部署流程；官方给出 RetinaNet-R50-FPN 及 ONNX CPU 结果。 | **真正的检测器实现。** 不能把 RetinaNet 结果外推到 YOLO11；完整接入需要适配 OpenMMLab 训练、图追踪与部署流程，适合作为第二候选或参考实现。 |
| [Microsoft NNI ModelSpeedup](https://github.com/microsoft/nni/blob/master/docs/source/tutorials/pruning_quick_start.py) | 先产生掩码，再由 `ModelSpeedup` 做图传播与层替换，将可改写的结构化掩码变成小张量；仓库还包含 Taylor 等剪枝器。 | **实现参考，不作新依赖。** [仓库](https://github.com/microsoft/nni)于 2024-09-18 归档，不能假设兼容当前 PyTorch/Ultralytics。 |
| [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) | 有 L1 和 FPGM 结构化滤波器裁剪；[FPGM 原论文](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Filter_Pruning_via_Geometric_Median_for_Deep_Convolutional_Neural_Networks_CVPR_2019_paper.html)及[作者代码](https://github.com/he-y/filter-pruning-geometric-median)按滤波器冗余而非仅按幅值找候选。 | Paddle 训练生态下的独立选择；不是直接读 Ultralytics `.pt` 的后端。可先移植评分思路，不必换框架；对依赖组要重新验证。 |
| [yolov5prune 社区实现](https://github.com/midasklr/yolov5prune/blob/v5.0/prune.py) | 针对早期 YOLOv5，按 [Network Slimming](https://arxiv.org/abs/1708.06519) 路线做 BN 稀疏训练，按系数挑通道，再重建小模型并迁移权重。 | **不依赖 Torch-Pruning 的 YOLO 特定参考**，不能直接套 YOLO11；不要把正常训练得到的小 BN 系数等同于经稀疏训练的结果。 |
| [社区 yolo-prune](https://github.com/xuexiren/yolo-prune) | 仓库声明为 YOLO11/YOLO26 的 `C3k2`、`C2PSA` 等块处理通道依赖，执行层仍是 Torch-Pruning。 | 是 **YOLO11 适配参考，不是独立执行引擎**；其兼容性和结果未用本地权重验证，不直接作为产品依赖。 |

## 按目标设备或精度决定剪枝预算

| 论文与参考实现 | 值得借鉴什么 | 主要限制 |
| --- | --- | --- |
| [NetAdapt，ECCV 2018](https://arxiv.org/abs/1804.03230) · [作者 PyTorch 重实现](https://github.com/denru01/netadapt) | 用目标平台的直测指标指导逐步收缩各层，依据短训后表现选择候选。比预设“每层 10%”更贴近部署目标。 | 是搜索策略参考，不是可直接替换的 YOLO11 库；评估和微调有成本。 |
| [HALP，NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/file/5434be94e82c54327bb9dcaf7fca52b6-Paper-Conference.pdf) · [NVlabs 官方代码](https://github.com/NVlabs/HALP/blob/main/README.md) | 在目标设备建立延迟查表，用通道重要性与延迟预算求解结构化剪枝；能识别某些宽度变化几乎没有延迟收益。 | 原仓库示例环境较旧，代码注明非商用研究/评估；不直接打包。查表只做预筛，最终仍以整模部署为准。 |
| [MDP，CVPR 2025](https://mdp-cvpr.github.io/) · [NVlabs 官方代码](https://github.com/NVlabs/MDP) | 联合选择通道、块、注意力维度等结构，用设备查表处理交互；可参考其延迟约束建模。 | 不是 YOLO11 通用适配器；复现按已发布模型与各子目录许可判断。首版不必引入其求解器。 |
| [EagleEye，ECCV 2020](https://arxiv.org/abs/2007.02491) · [作者代码](https://github.com/anonymous47823493/EagleEye) | 对候选子网重校准 BN 再评估，减少为每个候选进行完整微调的开销。 | BN 重校准需代表性训练数据，不用验证集调 BN；在象棋检测器上的候选排序可靠性待验证。 |
| [LeGR，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Chin_Towards_Efficient_Model_Compression_via_Learned_Global_Ranking_CVPR_2020_paper.html) · [作者代码](https://github.com/enyac-group/LeGR) | 学习跨层滤波器排名，而不是直接把不同层的原始幅值混起来比较。 | 需要训练或搜索且原代码包含架构专用逻辑；首版先测逐层敏感度，不直接迁整个框架。 |

[Ultralytics 的 YOLOv5 官方稀疏教程](https://docs.ultralytics.com/yolov5/tutorials/model-pruning-and-sparsity)是非结构化置零，不是物理删通道；[PyTorch 2:4](https://docs.pytorch.org/tutorials/advanced/semi_structured_sparse.html)是面向支持内核的半结构化格式，不等价于普通 YOLO 卷积缩宽。另一个易混淆点：[OpenVINO 2026 版本说明](https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html)记录 NNCF 已移除旧的结构化剪枝路径；不要把现在的 `nncf.prune()` 非结构化能力当作旧 filter pruning 的原位替代。此前调研中 NNCF 的静态量化推荐不受影响。

## 对本地象棋 YOLO11s 的验证顺序

1. **先确认目标后端和热点。** 用原始模型测纯推理与端到端，找出真正耗时的阶段。此前 PyTorch CUDA 832、batch 1 的对照里，默认物理剪枝只减约 3.03% 参数，mAP50–95 掉 7.29 个百分点，P50 无明显收益；第 3 层单剪就明显掉点，但这不能证明它独自解释了组合掉点。
2. **两条路线并跑。** 对照 A：保留现有依赖执行层，把候选选择换成 HALP/NetAdapt 式搜索；按实测宽度候选筛掉不省时间的组，再用验证 mAP 约束剪枝。对照 B：用 ModelOpt FastNAS 做独立小样；先验证原始图能转换、候选确实缩宽且可保存/重载/导出，才跑搜索。若图不兼容，记录失败节点，不默默回退成置零。
3. **以实际结构和部署结果验收。** [YOLO11 官方 YAML](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11.yaml)含 `C3k2`、`C2PSA`、多尺度拼接和 Detect。要核对分割、分支、注意力形状约束，以及检测类别/回归输出未被误改。组合剪后可能与逐层结果不同，必须重新测；直接比较剪后不微调表现，并另做统一微调预算下的对照，加入原生小版 YOLO11n。
4. **按精度与收益双门禁交付。** 搜索只用验证集；拆分外的 test 用于最终对照。梯度打分和 BN 校准用训练侧样本。记录 mAP50–95、类别 AP、参数量、产物大小与目标后端 P50/P95。同设备、同精度、同输入；如果没有候选同时满足精度预算和速度目标，输出“不建议剪枝”，不强制生成推荐模型。

## 覆盖范围与第二轮查漏

这是**面向工具选型的调研**，不是穷尽性文献综述。补查使用 CVF 论文索引、arXiv 作者论文、GitHub 仓库及目录。对 CVPR 2025/2026 公开索引做了“剪枝 × 通道/卷积/检测/延迟”标题筛查；这不包含摘要全文、所有会议、预印本和所有社区实现，不能据此判断该方向新工作的数量，也不能据此证明没有更合适的实现。

| 前一轮覆盖不足的分支 | 补查结论 |
| --- | --- |
| **按特征重建选通道** | [He 等，ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html)用 LASSO 选通道、最小二乘重建；[ThiNet，ICCV 2017](https://arxiv.org/abs/1707.06342)用下一层统计判断保留哪些滤波器。[He 作者仓库](https://github.com/ethanhe42/channel-pruning)和 [ThiNet 作者仓库](https://github.com/Roll920/ThiNet)是旧生态参考；不能直接跑现在的 YOLO11。它们提供了不同于幅值、Taylor 或 Fisher 的打分/选择对照。 |
| **用学习分配各层预算** | [AMC，ECCV 2018](https://hanlab.mit.edu/projects/amc)用强化学习搜索各层压缩率；[作者仓库](https://github.com/mit-han-lab/amc)提供 MobileNet/ImageNet 的搜索、导出和微调。[DSA，ECCV 2020](https://arxiv.org/abs/2004.02164)通过梯度学习各层稀疏率，[作者实现](https://github.com/walkerning/differentiable-sparsity-allocation)是分类实验参考，不是任意检测器后端。 |
| **同时搜索宽度与深度** | [DMS 论文](https://arxiv.org/abs/2405.07194)用可微 Top-k 搜索宽度/深度，报告过 YOLOv8n 在 COCO 上的检测结果；此次核对的[官方公开实验目录](https://github.com/LKJacky/Differentiable-Model-Scaling/tree/main/applications)只见 EfficientNet。因此它是值得跟踪的检测论文，不能列为已找到可复现 YOLO 配方的工具。 |
| **不同部署路径** | 物理删通道、带稀疏内核的 2:4、普通掩码置零是三种不同产物，不共用速度结论。整块删减和输入相关的动态跳过也不是固定通道剪枝；首版先不纳入，避免依赖不支持条件执行的部署后端。LLM、ViT token 剪枝及从零稀疏训练不在当前静态 YOLO11 部署范围。 |

**当前证据：** 只有现有实现和逐层消融在象棋 YOLO11s 上实测过。其他方案分别处于“论文或源码已核对”阶段；尚未证明能读取这份权重、真正缩窄、可重载导出，或在相同部署后端加速。分类或其他检测器的结果不可代替本机验证。

**选型结束条件：** 不要求补齐所有论文，但至少把现有方案、一个独立自动搜索实现、一个带目标设备延迟约束的策略，以及原生小模型在同一数据和后端下比较。记录版本、许可证、真实形状、导出结果、分类别 AP、重复测速的波动和无法运行的原因。未过兼容或收益门禁的候选保留失败记录，不标记为已支持。

## 近期检测剪枝：BN-γ 与替代路线

| 方法 | 新在哪里 | 复现边界 |
| --- | --- | --- |
| [HCF-YOLO，2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13766160/) | 是经典 BN-γ 在 YOLO11 上的近期应用，不是新评分公式。对不同层使用不同稀疏强度，保护浅层、高分辨率支路和回归输出，按各层 γ 分布选择宽度，最后微调。 | [复现标签内有 BN-γ 脚本](https://github.com/liyantao2025/HCF-YOLO/blob/full-reproducibility-v1/scripts/classic_bn_gamma_prune_final_egcg.py)，覆盖 HCF 自定义网络及 YOLO11n 对照。层号不可直接移植；论文中的 Backend/精度对比不是剪枝本身的同后端提速。 |
| [GSA-YOLO，2026](https://www.nature.com/articles/s41598-026-55599-2) | YOLOv8n 的颈部组稀疏、检测头结构选择及蒸馏。论文的通道缩放系数不是直接按原模型 BN-γ 剪。 | 公开 [SSS 正则脚本](https://github.com/KJHkong/GSA-YOLO/blob/main/scripts_GS/sss.py)和[剪枝脚本](https://github.com/KJHkong/GSA-YOLO/blob/main/scripts_GS/prune_sss.py)走的是卷积核 L1 路径，与论文描述的独立缩放系数存在差异；不能把这组脚本标为已复现论文配方。 |
| [位置感知检测器剪枝，WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Lan_Visual_Detector_Compression_via_Location-Aware_Discriminant_Analysis_WACV_2026_paper.html) | 用目标位置参与判别特征训练并沿网络追踪其贡献，按检测任务的重要性选结构，不依赖现成 γ。 | 核对时[作者仓库](https://github.com/lanqz7766/LDT_Pruning)只有说明和许可证，未找到训练/剪枝代码。先作为研究参考，不作为最近可跑的路线。 |

**象棋优先验证 HCF 式分层 BN-γ 稀疏训练。** 不照抄其层号和正则系数；用本模型的敏感度选择保护组。保留不加稀疏正则的同训练预算对照，并与 Taylor 在接近的实际参数缩减率、相同剪后微调预算和同部署后端下复测；尚未在象棋模型运行这套稀疏训练。
