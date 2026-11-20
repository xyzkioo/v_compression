"""Bridge PyTorch detection adapters into compression without a YOLO data contract."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .model_runtime import _call, _load_adapter, load_model, require_torch, resolve_device, save_model


class DetectionAdapter:
    GENERIC_STRUCTURED_METHODS = frozenset({"torch_pruning", "lamp", "taylor", "fisher"})

    def __init__(self, module: Any, config: Mapping[str, Any]) -> None:
        self.module = module
        self.config = config
        self.device = resolve_device(config["model"].get("device", "auto"))
        self.yolo = callable(getattr(module, "load_detector", None))

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> DetectionAdapter | None:
        spec = config.get("model", {}).get("adapter")
        if spec in (None, "", "torch", "builtin", "ultralytics"):
            return None
        return cls(_load_adapter(spec), config)

    def _hook(self, name: str):
        hook = getattr(self.module, name, None)
        if not callable(hook):
            raise ValueError(f"自定义检测 adapter 需要实现 {name}()。")
        return hook

    def load(self, weights: Path):
        torch = require_torch()
        if not weights.is_file():
            raise FileNotFoundError(f"找不到检测模型权重：{weights}")
        if self.yolo:
            detector = _call(self.module.load_detector, weights, device=self.device, config=self.config)
            if getattr(detector, "task", None) != "detect" or not isinstance(getattr(detector, "model", None), torch.nn.Module):
                raise TypeError("load_detector() 必须返回包含 PyTorch 模型的 YOLO 检测实例。")
            return detector
        model, _ = load_model({**self.config["model"], "weights": weights}, device=self.device)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("自定义检测 adapter 必须返回 torch.nn.Module。")
        return SimpleNamespace(model=model)

    def save(self, detector: Any, target: Path) -> None:
        if self.yolo:
            detector.save(str(target))
        else:
            save_model(detector.model, target, self.module)

    def evaluate(self, model: Any, run_dir: Path, name: str) -> dict[str, float]:
        torch = require_torch()
        output = run_dir / name
        output.mkdir(parents=True, exist_ok=True)
        model.to(self.device).eval()
        with torch.inference_mode():
            metrics = _call(
                self._hook("evaluate_detection"), model, config=self.config,
                split=str(self.config.get("evaluation", {}).get("split", "val")),
                device=self.device, run_dir=output,
            )
        if not isinstance(metrics, Mapping):
            raise TypeError("evaluate_detection() 必须返回检测指标字典。")
        result = {}
        for key in ("map50_95", "map50", "precision", "recall"):
            output_key = f"evaluation_{key}"
            value = metrics.get(output_key, metrics.get(key))
            if value is None and key != "map50_95":
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"evaluate_detection() 缺少有效指标 {key}。") from error
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError(f"evaluate_detection() 的 {key} 必须在 [0, 1] 内。")
            result[output_key] = number
        return result

    def make_forward(self, image_size: tuple[int, int], model: Any, *, pruning: bool = False):
        torch = require_torch()
        dtype = next(model.parameters()).dtype
        make_inputs = getattr(self.module, "make_inputs", None)
        sample = (_call(make_inputs, batch_size=1, image_size=image_size, device=self.device, dtype=dtype,
                        source=None, for_export=False)
                  if callable(make_inputs) else torch.zeros(1, 3, *image_size, device=self.device, dtype=dtype))
        if isinstance(sample, torch.Tensor):
            args, kwargs = (sample,), {}
        elif isinstance(sample, Mapping):
            args, kwargs = (), sample
        elif hasattr(sample, "args") and hasattr(sample, "kwargs"):
            args, kwargs = sample.args, sample.kwargs
        elif isinstance(sample, (tuple, list)):
            args, kwargs = sample, {}
        else:
            raise TypeError("make_inputs() 必须返回 Tensor、位置参数序列、关键字参数字典或 InputBundle。")
        trace = getattr(self.module, "forward_for_pruning", None)
        if pruning and callable(trace):
            return lambda current_model: trace(current_model, *args, **kwargs)
        return lambda current_model: current_model(*args, **kwargs)

    def pruning_targets(self, model: Any) -> list[str]:
        targets = _call(self._hook("pruning_targets"), model, config=self.config)
        if (not isinstance(targets, (list, tuple)) or not targets
                or any(not isinstance(name, str) or not name for name in targets)
                or len(set(targets)) != len(targets)):
            raise ValueError("pruning_targets() 必须返回至少一个互不重复的 Conv2d 层名。")
        return list(targets)

    def calibrate(self, model: Any, method: str) -> None:
        torch = require_torch()
        model.eval().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            _call(self._hook("calibrate_pruning"), model, method=method, config=self.config, device=self.device)
        if not any(param.grad is not None for param in model.parameters()):
            raise RuntimeError("calibrate_pruning() 未生成检测损失梯度。")

    def finetune(self, detector: Any, run_dir: Path, epochs: int):
        if not epochs:
            return detector, {"finetune_epochs": 0, "finetune": "skipped"}
        torch = require_torch()
        hook = self._hook("finetune_detection")
        model = detector.model.to(self.device).train().requires_grad_(True)
        result = _call(hook, model, epochs=epochs, config=self.config, device=self.device, run_dir=run_dir)
        if not isinstance(result, torch.nn.Module):
            raise TypeError("finetune_detection() 必须返回微调后的 torch.nn.Module。")
        return SimpleNamespace(model=result), {"finetune_epochs": epochs, "finetune": "completed"}
