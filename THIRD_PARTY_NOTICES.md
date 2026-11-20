# v_compression 第三方依赖说明

核对日期：2026-10-13。根目录 LICENSE 保留本项目原有 MIT 授权文本；该文本不重新授权第三方代码、库、模型或素材，也不证明全部代码的作者来源。实际发布时需保留使用版本要求的许可证和通知。

本项目业务代码负责：基线与阶段调度、配置校验、量化层统计、结构缩放和依赖感知剪枝接入、蒸馏训练流程、指标汇总、产物管理及来源记录。 模型计算、训练或优化的第三方底层能力不应表述为本项目自行创造的算法。

| 依赖 | 使用范围与许可来源 |
| --- | --- |
| PyTorch / torchvision | 模型加载、推理、训练及相关操作；各实际版本的 LICENSE/NOTICE 应随发布方式核对。[PyTorch 官方许可](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Ultralytics / 自定义 Fork | 支持的 YOLO 加载、预测、验证等；默认 AGPL-3.0 或另行取得的企业许可。Fork 还需检查自身附加代码的授权。[官方许可说明](https://www.ultralytics.com/license) |
| React / React DOM / lucide-react | 已构建前端包含的第三方组件；已复制当前安装版本的许可到 `licenses/frontend/`，版本与文件清单见该目录 `manifest.json`；随前端交付保留 |
| PySide6 / pywebview | 桌面窗口、浏览器桥接；按实际版本及打包方式保留其许可证/通知 |
| PyYAML / NumPy / Pillow / OpenCV | 配置、数值及图片处理；以安装版本的许可证和依赖通知为准 |

Ultralytics 官方提供 AGPL 与企业许可两条路径；根目录 MIT 文本不能替代其要求。当前文档修订没有购买企业许可，也没有自动将整个项目重新许可。是否为受相关义务约束的组合或衍生作品，应结合实际调用、修改和发布方式核对。

用户选择的权重、数据集、图片和外部 adapter 不属于本项目默认提供的原创资产。当前文件夹未复制 PyTorch/Ultralytics 的完整源码；从外部环境调用不等于许可证义务消失。依赖清单不是完整法律审计或已解决许可问题的证明。
