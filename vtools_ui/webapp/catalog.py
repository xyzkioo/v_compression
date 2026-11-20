"""Metadata for this project only."""
from pathlib import Path
from vtools_ui.standalone import identity
from model_compression.core.module_selection import COMPRESSION_MODES
from model_compression.core.pruning_methods import STRUCTURED_METHODS
ROOT = Path(__file__).resolve().parents[2]
APP = identity()
TOOLS = {'compression': {'title': '模型压缩',
                 'group': 'model',
                 'description': '选择一种处理模式，运行评估、剪枝、量化或蒸馏实验。',
                 'modes': COMPRESSION_MODES,
                 'structured_methods': STRUCTURED_METHODS,
                 'config': 'model_compression/config/config.yaml',
                 'entry': 'compression',
                 'fields': [{'key': 'task',
                             'label': '模型任务',
                             'kind': 'select',
                             'options': ['detect', 'classify'],
                             'default': 'detect'},
                            {'key': 'resource',
                             'label': '公共模型权重',
                             'kind': 'file',
                             'hint': '用于基线评估、剪枝或量化；检测蒸馏只需要教师和学生权重。'},
                            {'key': 'data',
                             'label': '检测数据集 YAML',
                             'kind': 'file',
                             'when': {'task': 'detect'},
                             'hint': 'YOLO 使用含 train/val 和 names 的 data.yaml。自定义检测 adapter 可以在工具配置的 dataset 节指定自己的数据集。'},
                            {'key': 'teacher_weights',
                             'label': '蒸馏教师权重',
                             'kind': 'file',
                             'when': {'task': 'detect'},
                             'hint': '检测蒸馏必填；教师类别名称及顺序须与数据集一致，需要支持原生蒸馏的 ultralytics-cn Fork。'},
                            {'key': 'student_weights',
                             'label': '蒸馏学生初始化权重',
                             'kind': 'file',
                             'when': {'task': 'detect'},
                             'hint': '可直接使用通用预训练小模型；自动按数据集适配检测头，无需预先训练，架构须与教师蒸馏特征层兼容。'},
                            {'key': 'train',
                             'label': '训练集目录',
                             'kind': 'dir',
                             'when': {'task': 'classify'},
                             'hint': '选择分类训练集根目录；下面按类别分子目录，例如 train/cat/*.jpg、train/dog/*.jpg。'},
                            {'key': 'val',
                             'label': '验证集目录',
                             'kind': 'dir',
                             'when': {'task': 'classify'},
                             'hint': '选择分类验证集根目录；下面按类别分子目录，例如 val/cat/*.jpg、val/dog/*.jpg。'},
                            {'key': 'device',
                             'label': '运行设备',
                             'kind': 'select',
                             'options': ['auto', 'cpu', 'cuda:0'],
                             'default': 'auto'}],
                 'modules': [('baseline.evaluate', '基线评估'),
                             ('compression.quantize.dynamic_int8', '动态 INT8'),
                             ('compression.quantize.onnx', '检测 ONNX 量化'),
                             ('compression.prune.unstructured', '非结构化剪枝'),
                             ('distillation.classification', '分类知识蒸馏'),
                             ('distillation.detection', '检测知识蒸馏'),
                             ('compression.prune.structured', '结构化通道缩放')]},
 'environment': {'title': '环境检查',
                 'group': 'system',
                 'description': '检查 Python、PyTorch、CUDA 和本地源码。',
                 'script': 'env_test/check_install.py',
                 'fields': [{'key': 'source',
                             'label': '源码目录（可选）',
                             'kind': 'dir',
                             'default': '',
                             'hint': '留空时自动查找 项目同级的 Ultralytics Fork。'},
                            {'key': 'weights', 'label': '模型权重', 'kind': 'file'},
                            {'key': 'yaml',
                             'label': '模型结构 YAML（可选）',
                             'kind': 'file',
                             'hint': '可选：已有 .pt 权重时留空；只有从模型结构创建网络时才选择包含 backbone 和 head 的 YAML。数据集 data.yaml '
                                     '请不要填在这里。'},
                            {'key': 'device',
                             'label': '运行设备',
                             'kind': 'select',
                             'options': ['auto', 'cpu', 'cuda:0'],
                             'default': 'auto'},
                            {'key': 'size', 'label': '合成输入尺寸', 'kind': 'number', 'default': '64'},
                            {'key': 'export', 'label': '检查 ONNX / TensorRT 导入', 'kind': 'bool'},
                            {'key': 'skip_model', 'label': '跳过模型构建和前向传播', 'kind': 'bool'}]}}

def public_catalog() -> dict[str, dict]:
    """Return JSON-safe metadata without server-side script paths."""
    result: dict[str, dict] = {}
    for key, value in TOOLS.items():
        item = {name: content for name, content in value.items() if name not in {"script", "entry"}}
        if "variants" in item:
            item["variants"] = {
                name: {
                    "title": variant["title"],
                    "defaults": variant.get("defaults", {}),
                    "fields": [
                        {"key": field, "label": label, "kind": kind, "required": required}
                        for field, label, kind, _flag, required in variant["fields"]
                    ],
                }
                for name, variant in item["variants"].items()
            }
        result[key] = item
    return result
