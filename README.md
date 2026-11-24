# v_compression 1.0.1

模型压缩项目，用于模型基线评估、量化、剪枝与知识蒸馏实验。

## 启动

在文件管理器中双击项目根目录的 **`v_compression`**。终端可使用：

```bash
./启动.sh
./启动.sh --browser
./启动.sh --check
```

`--browser` 在浏览器中打开界面；`--check` 检查本项目实际导入位置。启动失败时查看 `user-data/launcher.log`。移动时保留完整项目文件夹，外部权重、数据和解释器路径按新机器重新选择。

## 环境与输入

默认使用项目根目录的 `.venv/bin/python`；在界面“设置”中选择模型任务所用的解释器。界面依赖见 `requirements.txt`，模型功能还需对应的 PyTorch、Ultralytics 或其他 adapter 依赖。实际环境与重建方式见 [ENVIRONMENT.md](ENVIRONMENT.md)。

模型与数据放在 `inputs/`，或在界面选择外部文件。配置副本、任务记录、日志和产物默认写入本项目 `user-data/`。每次任务创建新的运行目录，输入权重和原始数据只读。

## 命令行

从本项目根目录执行：

```bash
./run-python.sh -m model_compression --help
./run-python.sh -m model_compression --request /path/to/request.json
```

接口字段与相对路径规则见 [接口说明](docs/interface.md)。项目运行已包含构建后的前端文件，正常启动不需要 Node；修改前端源码后需要重新构建。

## 项目文档

- [操作说明](docs/user-guide.md)
- [功能与验证范围](docs/function-scope.md)
- [视觉模型轻量化：论文、实现与首版路线](docs/lightweight-tool-research.md)
- [开发与维护记录](docs/development-record.md)
- [版本记录](VERSION.md)
- [第三方依赖与许可](THIRD_PARTY_NOTICES.md)

检测蒸馏使用教师、学生初始化权重和匹配的数据集。界面一次选择一种处理模式；检测蒸馏不要求额外填写公共模型权重。AMP 检查权重默认缓存到 `~/.cache/ultralytics/weights/`，可由 `XDG_CACHE_HOME` 更改缓存根目录。
