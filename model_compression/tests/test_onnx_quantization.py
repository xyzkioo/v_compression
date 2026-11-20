"""Regression tests for ONNX precision export and reload."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from model_compression.modules.onnx_quantization import export_onnx_precision


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
