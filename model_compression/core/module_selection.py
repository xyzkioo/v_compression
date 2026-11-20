"""模型压缩模块开关和命令行选择。"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


COMPRESSION_MODULES: dict[str, bool] = {
    "model.parameters": True,
    "baseline.evaluate": True,
    "compression.quantize.dynamic_int8": False,
    "compression.prune.unstructured": False,
    "compression.prune.structured": False,
    "distillation.classification": False,
    "distillation.detection": False,
    "comparison.report": False,
    "artifact.export": False,
}


COMPRESSION_MODES: dict[str, dict[str, Any]] = {
    'baseline': {'label': '基线评估', 'tasks': ['detect', 'classify'],
                 'description': '用原始权重统计模型参数量，在验证集测检测 mAP 或分类准确率。\n'
                                '对比压缩前后结果时，使用相同的验证集、输入尺寸和设备。',
                 'primary_module': None,
                 'modules': ['model.parameters', 'baseline.evaluate'], 'public_weights': True},
    'structured': {'label': '结构化剪枝', 'tasks': ['detect'],
                   'description': '对检测模型删除通道、缩短网络或重建较小规格，并按配置微调、评估。\n'
                                  '可勾选多种方法，每种都使用同一份输入权重。通用检测 adapter 可用幅值、LAMP、Taylor 和 Fisher，其他方法使用 YOLO 兼容路径。',
                   'primary_module': 'compression.prune.structured',
                   'modules': ['model.parameters', 'baseline.evaluate', 'compression.prune.structured'], 'public_weights': True},
    'unstructured': {'label': '非结构化剪枝', 'tasks': ['detect', 'classify'],
                     'description': '按权重绝对值给 Linear 和卷积层参与剪枝的权重排序，将全局最小的一部分置零。张量形状保持原样。\n'
                                    'sparsity 设置目标置零比例。看报告中的实际稀疏率和验证精度，普通稠密推理的耗时可能接近原模型。',
                     'primary_module': 'compression.prune.unstructured',
                     'modules': ['model.parameters', 'baseline.evaluate', 'compression.prune.unstructured'], 'public_weights': True},
    'detection_distillation': {'label': '检测知识蒸馏', 'tasks': ['detect'],
                               'description': '用教师指导较小的学生训练。教师类别和顺序需与数据集一致，学生检测头会按数据集自动适配。\n'
                                              '填写教师、学生权重与检测数据集 YAML，并使用支持检测蒸馏的 Ultralytics Fork。结果可对照学生训练前后的 mAP。',
                               'primary_module': 'distillation.detection',
                               'modules': ['distillation.detection'], 'public_weights': False},
    'quantization': {'label': '动态 INT8 量化', 'tasks': ['classify'],
                     'description': '将分类模型里受支持的 Linear、LSTM、GRU 等层转换为动态 INT8，推理时动态处理激活值。模型转换和评估在 CPU 上执行。\n'
                                    '主要适合含这些层的模型。报告列出实际量化层数、验证准确率和 CPU 推理耗时。',
                     'primary_module': 'compression.quantize.dynamic_int8',
                     'modules': ['model.parameters', 'baseline.evaluate', 'compression.quantize.dynamic_int8'], 'public_weights': True},
    'classification_distillation': {'label': '分类知识蒸馏', 'tasks': ['classify'],
                                    'description': '学生同时学习数据集的类别标签和教师给出的软预测。temperature 控制软预测的平滑程度，alpha 调整两部分损失的权重。\n'
                                                   '在 YAML 配置教师与学生，提供分类训练集和验证集。训练会保存学生检查点供后续评估。',
                                    'primary_module': 'distillation.classification',
                                    'modules': ['model.parameters', 'baseline.evaluate', 'distillation.classification'], 'public_weights': True},
}


def compression_mode_modules(mode: str, task: str) -> list[str]:
    definition = COMPRESSION_MODES.get(mode)
    if definition is None:
        raise ValueError(f'未知处理模式：{mode}')
    if task not in definition['tasks']:
        raise ValueError(f'处理模式「{definition["label"]}」不支持 {task} 任务。')
    return list(definition['modules'])


def standalone_detection_distillation(modules: Mapping[str, bool]) -> bool:
    allowed = {'model.parameters', 'baseline.evaluate', 'distillation.detection', 'comparison.report', 'artifact.export'}
    return bool(modules.get('distillation.detection')) and all(not enabled or key in allowed for key, enabled in modules.items())


def _split(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(item.strip() for item in str(value).split(",") if item.strip())
    return result


def resolve_compression_modules(
    config: Mapping[str, Any],
    *,
    only: Iterable[str] | None = None,
    enable: Iterable[str] | None = None,
    disable: Iterable[str] | None = None,
) -> dict[str, bool]:
    """按 YAML 和 CLI 解析模块。

    只要配置中存在 ``modules``，它就是显式 allow-list，省略的模块关闭，
    与 speed_test/model_diagnostics 的行为保持一致。
    """

    modules = dict(COMPRESSION_MODULES)
    configured = config.get("modules")
    if isinstance(configured, Mapping):
        unknown = set(map(str, configured)) - set(modules)
        if unknown:
            raise ValueError(f"未知模型压缩模块：{', '.join(sorted(unknown))}")
        modules = {key: False for key in modules}
        modules.update({str(key): bool(value) for key, value in configured.items()})
    elif configured is not None:
        raise ValueError("modules 必须是 module_id: true/false 的对象")

    only_values, enable_values, disable_values = (
        _split(only),
        _split(enable),
        _split(disable),
    )
    if only is not None and (enable_values or disable_values):
        raise ValueError("--only 不能与 --enable/--disable 同时使用")
    requested = set(only_values + enable_values + disable_values)
    unknown = requested - set(modules)
    if unknown:
        raise ValueError(f"未知模型压缩模块：{', '.join(sorted(unknown))}")
    overlap = set(enable_values) & set(disable_values)
    if overlap:
        raise ValueError(f"模块同时启用和禁用：{', '.join(sorted(overlap))}")
    # Preserve an explicitly empty selection instead of restoring YAML defaults.
    if only is not None:
        selected = set(only_values)
        modules = {key: key in selected for key in modules}
    else:
        modules.update({key: True for key in enable_values})
        modules.update({key: False for key in disable_values})
    return modules


__all__ = ["COMPRESSION_MODULES", "COMPRESSION_MODES", "compression_mode_modules",
           "standalone_detection_distillation", "resolve_compression_modules"]
