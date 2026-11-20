"""Register custom PyTorch classes and load their Ultralytics checkpoints."""

from __future__ import annotations

import importlib
from collections.abc import Mapping


def load_detector(weights, device, config):
    from ultralytics import YOLO

    settings = config["model"].get("adapter_options", {})
    if not isinstance(settings, Mapping):
        raise ValueError("model.adapter_options 必须是配置对象。")
    imports = settings.get("import_modules", [])
    required_layers = settings.get("required_layers", [])
    for key, values in (("import_modules", imports), ("required_layers", required_layers)):
        if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"model.adapter_options.{key} 必须是字符串列表。")
    for module in imports:
        importlib.import_module(module)
    detector = YOLO(str(weights), task="detect")
    for layer in required_layers:
        try:
            detector.model.get_submodule(layer)
        except AttributeError as error:
            raise ValueError(f"检测 checkpoint 缺少自定义层 {layer}：{weights}") from error
    detector.model.to(device).eval()
    return detector
