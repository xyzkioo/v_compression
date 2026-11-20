import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class LatencySearchTests(unittest.TestCase):
    def test_keeps_winner_only_after_accuracy_and_latency_gates(self):
        from torch import nn
        from model_compression.modules.latency_search import search_detector

        config = {"compression": {"structured": {"search_candidates": 1, "min_latency_gain": 0.05}}}
        for gain, accuracy, p95, accepted in ((0.08, 0.7, 9, True), (0.01, 0.7, 9, False),
                                              (0.08, 0.6, 9, False), (0.08, 0.7, 11, False)):
            with self.subTest(gain=gain, accuracy=accuracy, p95=p95), TemporaryDirectory() as directory:
                model = nn.Module()
                model.model = nn.Sequential(nn.Sequential(nn.Linear(4, 4)))
                original = SimpleNamespace(model=model)
                root = Path(directory)
                benchmark = lambda baseline, reduced: {"paired_gain": gain, "baseline_p95": 10, "candidate_p95": p95}
                with patch("model_compression.modules.latency_search._stage_candidates", return_value=[]), \
                        patch("model_compression.modules.latency_search._residual_blocks", return_value=["model.0.0"]):
                    def search():
                        return search_detector(original, config, root, 0.7, None,
                                               lambda candidate, name: {"evaluation_map50_95": accuracy}, benchmark)

                    if accepted:
                        candidate, metadata = search()
                        self.assertIsInstance(candidate.model.model[0][0], nn.Identity)
                        self.assertEqual(metadata["parameter_reduction"], 1)
                    else:
                        with self.assertRaisesRegex(ValueError, "建议保留原模型"):
                            search()
                self.assertIsInstance(original.model.model[0][0], nn.Linear)
                report = json.loads((root / "latency-search.json").read_text(encoding="utf-8"))
                self.assertEqual(report["selected_for_recheck"], "search-1" if accepted else None)
