"""可运行的 TorchVision 风格检测 adapter，包含 COCO 评估、校准和微调。"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Mapping

from model_compression.core.model_runtime import load_factory
from model_compression.core.run_manager import write_json
from model_compression.modules.gradient_calibration import accumulate_detection_gradients


def build_model(weights, device, *, model_config=None):
    """加载原始权重或工具生成的剪枝权重。"""
    import torch

    payload = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(payload, torch.nn.Module):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("model"), torch.nn.Module):
        return payload["model"]
    factory = (model_config or {}).get("factory")
    if not isinstance(payload, Mapping) or not factory:
        raise ValueError("权重须是完整 PyTorch 模型，或提供 model.factory 以加载 state_dict。")
    model = load_factory(str(factory), model_config)
    model.load_state_dict(payload.get("state_dict", payload))
    return model


def load_model(config, device):
    return build_model(config["weights"], device, model_config=config)


class _CocoSplit:
    def __init__(self, config, split):
        from torchvision.datasets import CocoDetection

        dataset = config["dataset"]
        annotations = dataset.get(split)
        image_root = (dataset.get("image_roots") or {}).get(split) or dataset.get("root")
        if not annotations or not image_root:
            raise ValueError(f"请配置 dataset.{split} 的 COCO JSON 以及 dataset.root 或 dataset.image_roots.{split}。")
        root = Path(config["project"]["root"]) / Path(image_root).expanduser()
        try:
            self.source = CocoDetection(str(root), str(annotations))
        except ImportError as error:
            raise RuntimeError("COCO 检测 adapter 需要 pycocotools。") from error
        if not len(self.source):
            raise ValueError(f"COCO {split} 划分没有图片。")

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        import torch
        from torchvision.transforms.functional import to_tensor

        image, annotations = self.source[index]
        boxes, labels = [], []
        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue
            class_id = int(annotation["category_id"])
            x, y, width, height = annotation["bbox"]
            if class_id <= 0 or width <= 0 or height <= 0:
                raise ValueError("TorchVision 检测模板需要正的 COCO 类别 ID 和有效的标注框。")
            boxes.append([x, y, x + width, y + height])
            labels.append(class_id)
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(labels, dtype=torch.int64)}
        return to_tensor(image), target, self.source.ids[index]


def _batches(dataset, batch_size, *, indices=None, shuffle=False):
    from torch.utils.data import DataLoader, Subset

    if batch_size < 1:
        raise ValueError("检测 adapter 的 batch_size 必须为正整数。")
    source = dataset if indices is None else Subset(dataset, indices)
    return DataLoader(source, batch_size=batch_size, shuffle=shuffle, collate_fn=tuple)


def _loss(model, batch, device):
    import torch

    images, targets, _ = zip(*batch)
    losses = model(
        [image.to(device) for image in images],
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )
    if not isinstance(losses, Mapping) or not losses or any(
        not isinstance(value, torch.Tensor) or value.ndim != 0 for value in losses.values()
    ):
        raise TypeError("模型训练前向须返回非空的标量损失字典。")
    return sum(losses.values())


def evaluate_detection(model, config, split, device, run_dir):
    """使用 COCO bbox 协议计算 mAP50–95 和 mAP50。"""
    import torch

    dataset = _CocoSplit(config, split)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch in _batches(dataset, int(config.get("evaluation", {}).get("batch_size", 1))):
            images, _, image_ids = zip(*batch)
            outputs = model([image.to(device) for image in images])
            if not isinstance(outputs, (list, tuple)) or len(outputs) != len(image_ids):
                raise TypeError("检测模型须为每张输入图片返回一个预测字典。")
            for image_id, output in zip(image_ids, outputs):
                boxes = output["boxes"].detach().cpu()
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()
                if boxes.shape != (scores.numel(), 4) or scores.shape != labels.shape or scores.ndim != 1:
                    raise ValueError("检测模型的 boxes、scores 和 labels 维度不一致。")
                for (left, top, right, bottom), score, label in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
                    if (label not in dataset.source.coco.cats or not 0 <= score <= 1
                            or not all(map(math.isfinite, (left, top, right, bottom)))
                            or right < left or bottom < top):
                        raise ValueError("预测须使用 COCO 类别 ID、有效分数和原图像素坐标的 xyxy 框。")
                    predictions.append({"image_id": image_id, "category_id": int(label), "score": score,
                                        "bbox": [left, top, right - left, bottom - top]})

    ground_truth = dataset.source.coco
    ground_truth.dataset.setdefault("info", {})
    if predictions:
        detections = ground_truth.loadRes(predictions)
    else:
        detections = COCO()
        detections.dataset = {**ground_truth.dataset, "annotations": []}
        detections.createIndex()
    evaluation = COCOeval(ground_truth, detections, "bbox")
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    metrics = {"map50_95": float(evaluation.stats[0]), "map50": float(evaluation.stats[1])}
    if min(metrics.values()) < 0:
        raise ValueError(f"COCO {split} 划分没有可用于计算 bbox mAP 的标注。")
    write_json(Path(run_dir) / "metrics.json", {**metrics, "split": split, "images": len(dataset),
                                               "predictions": len(predictions)})
    return metrics


def pruning_targets(model, config) -> list[str]:
    """读取显式配置的骨干内部卷积，顺序用于阶段选择。"""
    targets = config.get("compression", {}).get("structured", {}).get("pruning_targets")
    if (not isinstance(targets, list) or not targets
            or any(not isinstance(name, str) or not name.startswith("backbone.") for name in targets)):
        raise ValueError("请在 compression.structured.pruning_targets 指定可剪的骨干内部卷积层名。")
    return targets


def make_inputs(batch_size, image_size, device, dtype):
    """返回模型唯一的位置参数 images。"""
    import torch

    return ([torch.zeros(3, *image_size, device=device, dtype=dtype) for _ in range(batch_size)],)


def forward_for_pruning(model, images):
    """追踪骨干并校验全部输出特征形状，完整模型前向仍用于测速和评估。"""
    import torch

    return model.backbone(torch.stack(images))


def calibrate_pruning(model, method, config, device):
    """用标注训练样本计算 Taylor 或逐样本 Fisher 梯度。"""
    import torch

    section = config["compression"]["structured"]
    samples = int(section.get("calibration_images", 24))
    if samples < 1:
        raise ValueError("calibration_images 必须为正整数。")
    dataset = _CocoSplit(config, "train")
    indices = random.Random(42).sample(range(len(dataset)), min(len(dataset), samples))
    batch_size = 1 if method == "fisher" else int(section.get("batch_size", 4))
    training_modes = {layer: layer.training for layer in model.modules()}
    model.train()
    for layer in training_modes:
        if isinstance(layer, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.modules.dropout._DropoutNd)):
            layer.eval()
    try:
        batches = _batches(dataset, batch_size, indices=indices)
        accumulate_detection_gradients(model, method, batches, lambda batch: _loss(model, batch, device))
    finally:
        for layer, training in training_modes.items():
            layer.training = training


def finetune_detection(model, epochs, config, device, run_dir):
    """在剪后结构上用 AdamW 微调，并记录每轮训练损失。"""
    import torch

    section = config["compression"]["structured"]
    learning_rate = float(section.get("learning_rate", 0.001))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("检测微调学习率必须为有限的正数。")
    dataset = _CocoSplit(config, "train")
    batches = _batches(dataset, int(section.get("batch_size", 2)), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, batch, device)
            if not torch.isfinite(loss).item():
                raise ValueError(f"检测微调第 {epoch} 轮产生非有限损失。")
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
        history.append({"epoch": epoch, "loss": total / len(batches)})
        write_json(Path(run_dir) / "adapter-finetune.json", history)
    model.zero_grad(set_to_none=True)
    return model.eval()
