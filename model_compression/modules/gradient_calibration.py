"""Compute channel-group gradients from labelled detection images."""

from pathlib import Path
import random
from typing import Any


def accumulate_detection_gradients(model, method, batches, loss_for_batch):
    """Accumulate Taylor gradients or per-sample Fisher RMS gradients."""
    import torch

    if method not in {"taylor", "fisher"}:
        raise ValueError(f"梯度校准不支持方法 {method}。")
    squares = {}
    count = 0
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        for batch in batches:
            loss = loss_for_batch(batch)
            if loss.ndim != 0 or not torch.isfinite(loss).item():
                raise ValueError("检测校准损失必须是有限的标量。")
            loss.backward()
            count += 1
            if method == "fisher":
                for parameter in model.parameters():
                    if parameter.grad is None:
                        continue
                    energy = parameter.grad.detach().square()
                    if parameter in squares:
                        squares[parameter].add_(energy)
                    else:
                        squares[parameter] = energy
                model.zero_grad(set_to_none=True)
    for parameter, energy in squares.items():
        parameter.grad = energy.div_(count).sqrt_()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("训练校准没有生成检测损失梯度。")


def detection_calibrator(data: Path, size: int, samples: int):
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    if samples < 1:
        raise ValueError("calibration_images 必须为正整数。")
    dataset_config = check_det_dataset(str(data))
    cfg = get_cfg(overrides={"imgsz": size, "task": "detect", "cache": False})
    dataset = build_yolo_dataset(cfg, dataset_config["train"], 1, dataset_config, mode="val")
    if not len(dataset):
        raise ValueError("训练集没有可用的校准图片。")
    indices = random.Random(42).sample(range(len(dataset)), min(samples, len(dataset)))

    def calibrate(model: Any, method: str) -> None:
        size_per_batch = 1 if method == "fisher" else 4
        batches = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, indices), batch_size=size_per_batch, collate_fn=dataset.collate_fn,
        )
        device = next(model.parameters()).device
        previous_args = getattr(model, "args", None)
        model.args = cfg
        model.eval().requires_grad_(True)

        def loss_for_batch(batch):
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            batch["img"] = batch["img"].float() / 255
            loss, _ = model.loss(batch)
            return loss.sum()

        try:
            accumulate_detection_gradients(model, method, batches, loss_for_batch)
        finally:
            model.args = previous_args
            if hasattr(model, "criterion"):
                del model.criterion

    return calibrate, len(indices)
