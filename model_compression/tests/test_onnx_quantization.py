"""Regression tests for ONNX precision export and reload."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from model_compression.core.detection import run_detection_operation
from model_compression.modules.onnx_quantization import export_onnx_precision
from model_compression.run_model_compression import _RunState


class OnnxQualityGateTests(unittest.TestCase):
    def test_quantization_registers_only_candidates_that_pass_quality_gate(self):
        for score, expected in ((0.68, True), (0.60, False), (float("nan"), False)):
            with self.subTest(candidate_map=score), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                weights = root / "source.pt"
                data = root / "data.yaml"
                reference = root / "model-fp32.onnx"
                candidate = root / "model-int8.onnx"
                for path in (weights, data, reference, candidate):
                    path.write_bytes(b"test artifact")
                config = {
                    "model": {"task": "detect", "weights": str(weights)},
                    "dataset": {"data": str(data)},
                    "compression": {"quantization": {"max_map50_95_drop": 0.05}},
                }
                ultralytics = ModuleType("ultralytics")
                def detector_with_map(value):
                    box = SimpleNamespace(map=value, map50=0.9, mp=0.8, mr=0.8)
                    return SimpleNamespace(val=Mock(return_value=SimpleNamespace(box=box)))

                ultralytics.YOLO = Mock(side_effect=[detector_with_map(0.7), detector_with_map(score)])
                state = _RunState()
                detector = SimpleNamespace(model=SimpleNamespace(parameters=lambda: []))
                with patch.dict(sys.modules, {"ultralytics": ultralytics}), patch(
                    "model_compression.core.detection.load_detector", return_value=detector,
                ), patch(
                    "model_compression.modules.onnx_quantization.export_onnx_precision",
                    return_value=(candidate, reference, {"method": "onnx_int8", "precision": "int8"}),
                ):
                    if expected:
                        result = run_detection_operation("compression.quantize.onnx", config, state, root, "run")
                        self.assertEqual(result["status"], "succeeded")
                    else:
                        with self.assertRaisesRegex(ValueError, "限数值|降幅"):
                            run_detection_operation("compression.quantize.onnx", config, state, root, "run")
                self.assertEqual(Path(state.current(weights)[1]), candidate if expected else weights)
                report = root / "artifacts" / "quantization" / "onnx-comparison.json"
                if math.isfinite(score):
                    self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["quality_gate"]["passed"], expected)
                else:
                    self.assertFalse(report.exists())


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("torch", "onnx", "onnxruntime")),
                     "requires the optional ONNX dependencies")
class OnnxPrecisionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "requires detection postprocessing")
    def test_threshold_sweep_counts_duplicate_and_wrong_class_boxes(self):
        import numpy as np
        import torch

        from model_compression.tests.deployment_probe import _threshold_sweep

        outputs = np.array([[
            [2, 2, 12, 12, 0.95, 0],
            [2, 2, 12, 12, 0.70, 0],
            [16, 16, 26, 26, 0.80, 0],
        ]], dtype=np.float32)
        targets = torch.tensor([[2, 2, 12, 12, 0], [16, 16, 26, 26, 1]], dtype=torch.float32)
        points = _threshold_sweep([[outputs]], [((32, 32), targets)], 32, end_to_end=True)

        self.assertEqual(len(points), 6)
        for point in points:
            self.assertEqual((point["tp"], point["fp"], point["fn"]), (1, 2, 1))
            self.assertAlmostEqual(point["precision"], 1 / 3)
            self.assertEqual(point["recall"], 0.5)

    def test_fp16_casts_are_sorted_and_float_io_is_reloaded(self):
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "source.pt"
            weights.write_bytes(b"original")
            inputs = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 32, 32])
            outputs = helper.make_tensor_value_info("outputs", TensorProto.FLOAT, [1, 2, 32, 32])
            tensor = numpy_helper.from_array(np.full((2, 3, 1, 1), 0.25, dtype=np.float32), "weight")
            graph = helper.make_graph(
                [helper.make_node("Conv", ["images", "weight"], ["outputs"], name="conv")],
                "detector", [inputs], [outputs], [tensor],
            )
            original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
            helper.set_model_props(original, {"task": "detect"})

            def load_detector(checkpoint):
                def export(**_options):
                    target = checkpoint.with_suffix(".onnx")
                    onnx.save(original, target)
                    return target

                stride = SimpleNamespace(max=lambda: SimpleNamespace(item=lambda: 32))
                return SimpleNamespace(model=SimpleNamespace(stride=stride), export=export)

            with patch("model_compression.modules.onnx_quantization._preserve_custom_yolo_forward"):
                target, reference, report = export_onnx_precision(
                    weights, load_detector, root / "unused.yaml", root / "quantization",
                    {"dataset": {"input_size": 32}, "compression": {"quantization": {
                        "precision": "fp16", "cpu_threads": 1, "benchmark_warmup": 0, "benchmark_repeats": 1,
                    }}},
                )
            converted = onnx.load(target)
            onnx.checker.check_model(converted)
            self.assertEqual(converted.graph.input[0].type.tensor_type.elem_type, TensorProto.FLOAT)
            self.assertEqual(converted.graph.output[0].type.tensor_type.elem_type, TensorProto.FLOAT)
            self.assertEqual({item.key: item.value for item in converted.metadata_props}, {"task": "detect"})
            self.assertEqual(report["fp16_initializer_count"], 1)
            self.assertEqual(reference.name, "model-fp32.onnx")
            self.assertEqual(weights.read_bytes(), b"original")
            self.assertFalse(weights.with_suffix(".onnx").exists())
