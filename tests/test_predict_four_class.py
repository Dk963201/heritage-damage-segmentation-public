"""Inference correctness tests. Synthetic/random models give no research evidence."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from train_four_class import CLASSES, MODELS, evaluate
from predict_four_class import (load_model, load_roi, main, overlay_image,
                                predict_logits, prediction_arrays, run, sha256,
                                validate_protocol, validate_roi, verify_checkpoint_protocol_association)


def protocol(**overrides):
    return {"models": ["unet"], "classes": CLASSES, "patch": 32, "no_clahe": True, **overrides}


class TileSensitiveModel(torch.nn.Module):
    def forward(self, image):
        _, _, height, width = image.shape
        xx = torch.linspace(-2, 2, width)[None, None, :].expand(image.shape[0], height, width)
        yy = torch.linspace(-1, 1, height)[None, :, None].expand(image.shape[0], height, width)
        return torch.stack((image[:, 0] + xx, image[:, 1] - xx, image[:, 2] + yy, -yy), dim=1)


class FourClassInferenceTests(unittest.TestCase):
    def test_prediction_parity_with_training_evaluate_edges_overlap_and_padding(self):
        rng = np.random.default_rng(17)
        for height, width, clahe in [(47, 45, False), (47, 45, True), (13, 17, False)]:
            with self.subTest(shape=(height, width), clahe=clahe), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
                image.save(root / "image.png")
                logits = predict_logits(TileSensitiveModel(), image, 32, clahe)
                mask, probability, summary, valid = prediction_arrays(logits)
                self.assertEqual(mask.shape, (height, width))
                self.assertEqual(probability.shape, (4, height, width))
                self.assertTrue(np.isfinite(probability).all())
                np.testing.assert_allclose(probability.sum(axis=0), 1, atol=1e-6)
                np.testing.assert_array_equal(probability.argmax(axis=0), mask)
                self.assertEqual(sum(row["pixels"] for row in summary), height*width)
                self.assertAlmostEqual(sum(row["fraction"] for row in summary), 1)
                # Use independent trainer evaluation to compare every predicted pixel.
                Image.fromarray(mask).save(root / "mask.png")
                result = evaluate(TileSensitiveModel(), [{"image_path": "image.png", "mask_path": "mask.png"}],
                                  root, 32, clahe)
                confusion = np.array(result["confusion_matrix"])
                self.assertEqual(int(np.trace(confusion)), height*width)
                self.assertEqual(int(confusion.sum()), height*width)

    def test_probabilities_are_softmax_of_averaged_logits(self):
        logits = np.array([[[1000, 2]], [[999, 0]], [[998, -2]], [[997, 1]]], np.float32)
        mask, probabilities, _, _ = prediction_arrays(logits)
        expected = torch.softmax(torch.from_numpy(logits), dim=0).numpy()
        np.testing.assert_allclose(probabilities, expected, atol=1e-7)
        np.testing.assert_array_equal(mask, [[0, 0]])

    def test_roi_counts_exclusion_and_overlay_alignment(self):
        logits = np.zeros((4, 7, 9), np.float32)
        logits[2] = 2
        roi = np.zeros((7, 9), np.uint8)
        roi[1:5, 2:7] = 255
        mask, probability, summary, valid = prediction_arrays(logits, roi)
        self.assertEqual(summary[2]["pixels"], 20)
        self.assertEqual(summary[2]["fraction"], 1)
        self.assertTrue(np.all(mask[~valid] == 255))
        self.assertTrue(np.isnan(probability[:, ~valid]).all())
        np.testing.assert_allclose(probability[:, valid].sum(axis=0), 1, atol=1e-6)
        source = Image.fromarray(np.arange(7*9*3, dtype=np.uint8).reshape(7, 9, 3))
        overlay = overlay_image(source, mask, valid)
        self.assertEqual(overlay.size, source.size)
        np.testing.assert_array_equal(np.array(overlay)[~valid], np.array(source)[~valid])

    def test_invalid_roi_protocol_input_and_model_outputs(self):
        for roi in [np.ones((3, 2), np.uint8), np.full((2, 2), 2, np.uint8),
                    np.zeros((2, 2), np.uint8), np.ones((2, 2), np.float32),
                    np.array([[0, 1], [255, 0]], np.uint8)]:
            with self.assertRaises(ValueError):
                validate_roi(roi, (2, 2))
        for invalid in [protocol(patch=33), protocol(patch=True), protocol(classes=CLASSES[::-1]),
                        protocol(classes=CLASSES[:3]), protocol(no_clahe="false"),
                        protocol(encoder="resnet50"), protocol(num_classes=2), protocol(models=["fpn"]),
                        protocol(in_channels=1)]:
            with self.assertRaises(ValueError):
                validate_protocol(invalid, "unet")
        with self.assertRaisesRegex(ValueError, "RGB"):
            predict_logits(TileSensitiveModel(), Image.new("L", (32, 32)), 32, False)
        class Wrong(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 1, 32, 32)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            predict_logits(Wrong(), Image.new("RGB", (32, 32)), 32, False)
        with self.assertRaises(ValueError):
            prediction_arrays(np.full((4, 2, 2), np.nan))
        with self.assertRaises(ValueError):
            prediction_arrays(np.zeros((4, 0, 2), np.float32))
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "roi.png"
            Image.new("RGB", (2, 2)).save(p)
            with self.assertRaisesRegex(ValueError, "single-channel"):
                load_roi(p, (2, 2))

    def test_weights_only_rejects_wrong_architecture_and_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.pt"
            torch.save({"unrelated.weight": torch.zeros(4, 3)}, path)
            with self.assertRaisesRegex(ValueError, "architecture"):
                load_model(path, protocol(), "unet")
            with self.assertRaisesRegex(ValueError, "unavailable"):
                load_model(Path(directory) / "missing.pt", protocol(), "unet")
            with self.assertRaisesRegex(ValueError, "local"):
                load_model("https://example.invalid/model.pt", protocol(), "unet")

    def test_sidecar_verifies_pairing_and_rejects_swapped_preprocessing_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, settings = root / "unet-best.pt", root / "protocol.json"
            checkpoint.write_bytes(b"sidecar association fixture, not model weights")
            settings.write_text(json.dumps(protocol()))
            sidecar = checkpoint.with_suffix(".metadata.json")
            association = {"checkpoint_sha256": sha256(checkpoint), "protocol_sha256": sha256(settings),
                           "architecture": "unet", "num_classes": 4, "scope": "SYNTHETIC SOFTWARE TEST ONLY"}
            sidecar.write_text(json.dumps(association))
            result = verify_checkpoint_protocol_association(checkpoint, settings, "unet")
            self.assertEqual(result["status"], "association verified by matching hashes")
            # Same architecture/classes but different CLAHE must fail BEFORE model loading.
            wrong_settings = root / "different-preprocessing.json"
            wrong_settings.write_text(json.dumps(protocol(no_clahe=False)))
            image = root / "image.png"
            Image.new("RGB", (32, 32)).save(image)
            with patch("predict_four_class.load_model") as load:
                with self.assertRaisesRegex(ValueError, "protocol_sha256 mismatch"):
                    run(checkpoint, wrong_settings, "unet", image, root / "prediction")
                load.assert_not_called()
            for field, wrong in [("architecture", "fpn"), ("num_classes", 2), ("checkpoint_sha256", "0"*64)]:
                sidecar.write_text(json.dumps({**association, field: wrong}))
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    verify_checkpoint_protocol_association(checkpoint, settings, "unet")

    def test_random_checkpoint_cli_smoke_software_only_and_no_overwrite(self):
        # Random weights test loading/inference/output only, not training or accuracy.
        torch.manual_seed(6)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, settings, image_path = root / "random.pt", root / "protocol.json", root / "image.png"
            with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("No downloads allowed")):
                model = MODELS["unet"](encoder_name="efficientnet-b0", encoder_weights=None, classes=4)
                torch.save(model.state_dict(), checkpoint)
                del model
                settings.write_text(json.dumps(protocol(patch=64)))
                image = np.random.default_rng(9).integers(0, 256, (45, 37, 3), dtype=np.uint8)
                Image.fromarray(image).save(image_path)
                roi = np.zeros((45, 37), np.uint8)
                roi[3:40, 4:32] = 255
                roi_path = root / "roi.png"
                Image.fromarray(roi).save(roi_path)
                output = root / "result"
                argv = ["--checkpoint", str(checkpoint), "--protocol", str(settings), "--model", "unet",
                        "--image", str(image_path), "--roi", str(roi_path), "--output", str(output)]
                with contextlib.redirect_stdout(io.StringIO()):
                    main(argv)
                metadata = json.loads((output / "metadata.json").read_text())
                self.assertEqual(metadata["class_labels"][1], "Visible crack")
                self.assertEqual(metadata["included_pixels"], int((roi > 0).sum()))
                self.assertIn("NOT ASSIGNED", metadata["severity"])
                self.assertEqual(metadata["checkpoint_protocol_association"]["status"], "association unverified")
                probabilities = np.load(output / "prediction-probabilities.npy", allow_pickle=False)
                mask = np.array(Image.open(output / "prediction-classes.png"))
                overlay = np.array(Image.open(output / "prediction-overlay.png"))
                self.assertEqual(mask.shape, image.shape[:2])
                np.testing.assert_array_equal(mask[roi == 0], 255)
                np.testing.assert_array_equal(overlay[roi == 0], image[roi == 0])
                np.testing.assert_allclose(probabilities[:, roi > 0].sum(axis=0), 1, atol=1e-6)
                before = (output / "prediction-classes.png").read_bytes()
                with self.assertRaisesRegex(ValueError, "fresh"):
                    run(checkpoint, settings, "unet", image_path, output, roi_path)
                self.assertEqual(before, (output / "prediction-classes.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
