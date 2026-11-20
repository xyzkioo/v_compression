# 模型压缩项目的模型环境检查

从本项目根目录执行；使用与模型任务相同的 Python 解释器：

```bash
python env_test/check_install.py --help
python env_test/check_install.py --device cpu --expected-version none
```

使用自定义 Ultralytics 源码时，显式指定外部目录：

```bash
python env_test/check_install.py --source /path/to/ultralytics-fork --device cpu --expected-version none
python env_test/check_install.py --source /path/to/ultralytics-fork --weights /path/to/best.pt --device cpu --expected-version none
```

`--source` 指向包含 `ultralytics/` 的项目根目录；不要求目录与本项目相邻，也不会安装或修改该源码。包导入名通常仍为 `ultralytics`。权重依赖自定义层时，应选择训练时使用的兼容源码。

检查会报告解释器、真实导入来源、依赖一致性、版本、CUDA 可用性，并按参数尝试模型构建或合成图片预测。CPU 可以运行基础检查，CUDA/TensorRT 需要对应硬件与运行时。

| 参数 | 用途 |
| --- | --- |
| `--weights PATH` | 加载已有权重并尝试合成图片预测 |
| `--yaml PATH` | 指定模型结构 YAML；不能填写数据集 YAML |
| `--device auto/cpu/cuda:0` | 选择检查设备 |
| `--input-size 64` | 指定合成输入尺寸 |
| `--skip-model` | 只检查导入与依赖，不构建模型 |
| `--check-export` | 检查 ONNX/TensorRT 导入，不代表完成真实导出或测速 |
| `--no-local-check` | 不要求来源为本地 Fork |
| `--expected-version none` | 跳过针对特定 Fork 的预设版本比较 |
| `--strict` | 把版本、依赖和可选组件问题也计为失败 |

启动图形界面请双击项目根目录的 `v_compression`，或执行 `./启动.sh`。在“设置”中填写“模型任务使用的 Python 解释器”，并保存。每个项目使用自己的源码、配置和运行记录；第三方模型环境可显式选择。

检查成功只证明该环境通过所执行的检查，不证明所有业务功能或所有模型兼容。实际验证范围见 [功能说明](../docs/function-scope.md)。
