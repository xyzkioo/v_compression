"""Regression coverage for custom detector checkpoints and output protection."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from model_compression.core.detection import run_detection_operation
from model_compression.core.model_runtime import _load_adapter
from model_compression.run_model_compression import _RunState


ADAPTER = """
import torch

class Detector(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Conv2d(1, 8, 1), torch.nn.ReLU(), torch.nn.Conv2d(8, 2, 1))

def build_model(weights, device):
    return torch.load(weights, map_location=device, weights_only=False)

def make_inputs(batch_size, image_size, device, dtype):
    return (torch.ones(batch_size, 1, *image_size, device=device, dtype=dtype),)

def evaluate_detection(model, config, split, device, run_dir):
    with (run_dir / 'validation.txt').open('w') as stream:
        print(split, file=stream)
    return {'map50_95': 0.5}

def pruning_targets(model):
    return ['0']
"""


@unittest.skipUnless(importlib.util.find_spec("torch_pruning"), "torch-pruning is optional")
class DetectionAdapterTests(unittest.TestCase):
    def test_custom_baseline_pruning_and_reload_use_the_same_adapter(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "adapter.py"
            source.write_text(ADAPTER, encoding="utf-8")
            weights = root / "original.pt"
            torch.save(_load_adapter(str(source)).Detector(), weights)
            config = {
                "model": {"weights": str(weights), "adapter": str(source), "task": "detect", "device": "cpu"},
                "dataset": {"input_size": [3, 5]},
                "evaluation": {"split": "test"},
                "compression": {"structured": {
                    "method": "torch_pruning", "channel_sparsity": 0.25, "channel_round": 1,
                    "finetune_epochs": 0, "benchmark_warmup": 0, "benchmark_repeats": 1,
                }},
            }
            state = _RunState()
            run_dir = root / "run"
            native_loader = "model_compression.core.detection.load_detector"
            with patch(native_loader, side_effect=AssertionError("custom detector used YOLO")):
                baseline = run_detection_operation("baseline.evaluate", config, state, run_dir, "run")
                pruned = run_detection_operation("compression.prune.structured", config, state, run_dir, "run")
                config["compression"]["structured"]["method"] = "scale"
                with self.assertRaisesRegex(ValueError, "load_detector"):
                    run_detection_operation("compression.prune.structured", config, state, run_dir, "run")
            self.assertEqual(baseline["metadata"]["evaluation_map50_95"], 0.5)
            metrics = pruned["metadata"]
            self.assertEqual(metrics["map50_95_delta"], 0)
            self.assertEqual(metrics["source_parameter_count"], 34)
            self.assertEqual(metrics["structured_parameter_count"], 26)
            self.assertEqual(metrics["pruned_layers"][0]["layer_name"], "0")
            self.assertEqual(metrics["pruning_input_size"], [3, 5])
            for name in ("baseline_validation", "before_structured_pruning", "after_structured_pruning"):
                self.assertEqual((run_dir / name / "validation.txt").read_text().strip(), "test")
            self.assertTrue((run_dir / "artifacts" / "structured_pruning" / "detector-structured.pt").is_file())
