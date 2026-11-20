"""Exercise the concrete COCO adapter with a small TorchVision detector."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_compression.adapters import detection_adapter_template as adapter
from model_compression.core.detection import run_detection_operation
from model_compression.modules.gradient_calibration import accumulate_detection_gradients
from model_compression.run_model_compression import _RunState


def build_tiny_detector():
    import torch
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 12, 3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(12, 8, 3, padding=1),
    )
    backbone.out_channels = 8
    return FasterRCNN(
        backbone, min_size=32, max_size=32, rpn_anchor_generator=AnchorGenerator(((8, 16),), ((1.0,),)),
        rpn_pre_nms_top_n_train=16, rpn_pre_nms_top_n_test=12, rpn_post_nms_top_n_train=8,
        rpn_post_nms_top_n_test=6, box_detections_per_img=4, box_batch_size_per_image=8,
        box_roi_pool=MultiScaleRoIAlign(["0"], 3, 2),
        box_head=TwoMLPHead(8 * 3 * 3, 16), box_predictor=FastRCNNPredictor(16, 2),
    )


class DetectionCalibrationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_taylor_and_fisher_accumulate_distinct_gradients(self):
        import torch

        model = torch.nn.Linear(1, 1, bias=False)

        def loss(sample):
            return model(torch.tensor([[sample]])).sum()

        accumulate_detection_gradients(model, "taylor", [3.0, -3.0], loss)
        self.assertEqual(model.weight.grad.item(), 0)
        accumulate_detection_gradients(model, "fisher", [3.0, -3.0], loss)
        self.assertEqual(model.weight.grad.item(), 3)


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("torch", "torchvision", "pycocotools", "torch_pruning")),
    "requires the COCO adapter dependencies",
)
class DetectionAdapterTemplateTests(unittest.TestCase):
    def setUp(self):
        import torch
        from PIL import Image

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for split, image_ids in (("train", [1, 2]), ("val", [3, 4])):
            images, annotations = [], []
            for image_id in image_ids:
                name = f"{image_id}.png"
                with Image.new("RGB", (32, 32), color=(image_id * 40, 0, 0)) as image:
                    image.save(self.root / name)
                images.append({"id": image_id, "file_name": name, "width": 32, "height": 32})
                annotations.append({"id": image_id, "image_id": image_id, "category_id": 1,
                                    "bbox": [8, 8, 16, 16], "area": 256, "iscrowd": 0})
            (self.root / f"{split}.json").write_text(json.dumps({
                "images": images, "annotations": annotations, "categories": [{"id": 1, "name": "object"}],
            }), encoding="utf-8")
        self.config = {
            "project": {"root": str(self.root)},
            "model": {"task": "detect", "device": "cpu", "adapter": adapter.__name__},
            "dataset": {"root": str(self.root), "train": str(self.root / "train.json"),
                        "val": str(self.root / "val.json"), "input_size": [32, 32]},
            "compression": {"structured": {
                "pruning_targets": ["backbone.0"], "channel_sparsity": 0.25, "channel_round": 1,
                "calibration_images": 2, "batch_size": 2, "finetune_epochs": 1,
                "benchmark_warmup": 0, "benchmark_repeats": 1, "max_map50_95_drop": 1.0,
            }},
        }

    def test_coco_metrics_use_predictions_and_include_empty_results(self):
        import torch

        class FixedDetector(torch.nn.Module):
            def forward(self, images):
                boxes = torch.tensor([[8, 8, 24, 24]], dtype=torch.float32)
                scores = torch.tensor([0.99])
                labels = torch.tensor([1])
                return [{"boxes": boxes, "scores": scores, "labels": labels} for _ in images]

        model = FixedDetector()
        metrics = adapter.evaluate_detection(model, self.config, "val", "cpu", self.root)
        self.assertAlmostEqual(metrics["map50_95"], 1.0)
        self.assertAlmostEqual(metrics["map50"], 1.0)
        with patch.object(model, "forward", return_value=[
            {"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "labels": torch.empty(0, dtype=torch.long)},
        ]):
            empty = adapter.evaluate_detection(model, self.config, "val", "cpu", self.root)
        self.assertEqual(empty["map50_95"], 0)
        report = json.loads((self.root / "metrics.json").read_text())
        self.assertEqual(report["predictions"], 0)
        self.assertEqual(report["images"], 2)

    def test_four_methods_train_and_reload_with_real_torchvision_detector(self):
        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            original = build_tiny_detector()
        weights = self.root / "original.pt"
        torch.save({"state_dict": original.state_dict()}, weights)
        original_bytes = weights.read_bytes()
        self.config["model"].update(weights=str(weights), factory=f"{__name__}:build_tiny_detector")
        for method in ("torch_pruning", "lamp", "taylor", "fisher"):
            with self.subTest(method=method):
                self.config["compression"]["structured"]["method"] = method
                run_dir = self.root / method
                state = _RunState()
                baseline = run_detection_operation("baseline.evaluate", self.config, state, run_dir, method)
                result = run_detection_operation("compression.prune.structured", self.config, state, run_dir, method)
                metrics = result["metadata"]
                self.assertEqual(metrics["finetune"], "completed")
                self.assertLess(metrics["structured_parameter_count"], metrics["source_parameter_count"])
                self.assertAlmostEqual(metrics["map50_95_delta"],
                                       metrics["evaluation_map50_95"] - baseline["metadata"]["evaluation_map50_95"])
                reloaded = adapter.build_model(Path(result["artifact"]["path"]), "cpu")
                self.assertEqual(reloaded.backbone[0].out_channels, 9)
                self.assertEqual(reloaded.backbone[-1].out_channels, 8)
                self.assertEqual(reloaded.roi_heads.box_predictor.cls_score.out_features, 2)
                history = run_dir / "artifacts" / "structured_pruning" / "adapter-finetune.json"
                self.assertGreater(json.loads(history.read_text())[0]["loss"], 0)
        self.assertEqual(weights.read_bytes(), original_bytes)

    def test_pruning_cannot_change_the_backbone_interface(self):
        import torch

        weights = self.root / "original.pt"
        torch.save(build_tiny_detector(), weights)
        self.config["model"]["weights"] = str(weights)
        self.config["compression"]["structured"].update(method="torch_pruning", pruning_targets=["backbone.2"])
        run_dir = self.root / "invalid-target"
        with self.assertRaisesRegex(RuntimeError, "输出形状"):
            run_detection_operation("compression.prune.structured", self.config, _RunState(), run_dir, "invalid")
        self.assertFalse((run_dir / "artifacts" / "structured_pruning" / "detector-structured.pt").exists())
