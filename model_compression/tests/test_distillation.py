import tempfile
import unittest
from pathlib import Path


class DistillationTests(unittest.TestCase):
    def test_validation_loader_keeps_training_class_indices(self):
        from PIL import Image
        from model_compression.core.model_runtime import build_dataloaders

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in ("train/cat", "train/dog", "val/dog"):
                folder = root / relative
                folder.mkdir(parents=True)
                Image.new("RGB", (4, 4)).save(folder / "sample.png")
            config = {"dataset": {"train": root / "train", "val": root / "val"}, "input_size": [4, 4]}
            train, val = build_dataloaders(config)
            self.assertEqual(val.dataset.class_to_idx, train.dataset.class_to_idx)
            self.assertEqual(next(iter(val))[1].tolist(), [1])

            unknown = root / "val" / "bird"
            unknown.mkdir()
            Image.new("RGB", (4, 4)).save(unknown / "sample.png")
            with self.assertRaisesRegex(ValueError, "bird"):
                build_dataloaders(config)

    def test_loss_matches_weighted_ce_and_kl(self):
        import torch
        import torch.nn.functional as F

        from model_compression.modules.distillation import distillation_loss

        student = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        teacher = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
        labels = torch.tensor([1, 0])
        temperature, alpha = 2.0, 0.25
        expected = (1 - alpha) * F.cross_entropy(student, labels) + alpha * temperature**2 * F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1),
            reduction="batchmean",
        )
        actual = distillation_loss(student, teacher, labels, temperature=temperature, alpha=alpha)
        self.assertTrue(torch.allclose(actual, expected))

    def test_invalid_temperature_is_rejected(self):
        from model_compression.modules.distillation import validate_distillation_config

        with self.assertRaises(ValueError):
            validate_distillation_config({"temperature": 0})


if __name__ == "__main__":
    unittest.main()
