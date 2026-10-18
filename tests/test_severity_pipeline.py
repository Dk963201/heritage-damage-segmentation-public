"""Synthetic SOFTWARE-ONLY fixtures; these tests are not research evidence."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from severity_pipeline import (FIELDS, SeverityCNN, classification_metrics, evaluate,
                               load_checkpoint, object_sha, predict, preprocess, seed_everything,
                               sha, train, validate_config, validate_dataset, validate_rubric)


class SeverityWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rubric = json.loads((ROOT / "severity-rubric-draft.json").read_text())
        self.rubric.update(status="approved", version="1.0-test", approved_by="SOFTWARE TEST FIXTURE ONLY", approved_on="2026-01-01")
        self.rubric_path = self.root / "rubric.json"
        self.rubric_path.write_text(json.dumps(self.rubric))
        self.manifest = self.root / "manifest.csv"
        self.config = {"seed": 17, "epochs": 1, "batch_size": 4, "image_size": 96, "lr": 1e-4}
        self.rows = []
        rng = np.random.default_rng(17)
        # 24 distinct synthetic photos, both sites in each split and all 4 classes.
        for split in ("train", "validation", "test"):
            for site in ("Erbil", "Amedi"):
                for label in ("none", "low", "moderate", "high"):
                    ident = f"fixture-{split}-{site}-{label}"
                    path = self.root / f"{ident}.png"
                    Image.fromarray(rng.integers(0, 256, (72, 96, 3), dtype=np.uint8)).save(path)
                    self.rows.append({"image_id": ident, "image_path": path.name, "source_sha256": sha(path),
                                      "site": site, "building_group": f"fixture-building-{site}-{split}",
                                      "duplicate_group": ident, "split": split, "severity_label": label,
                                      "review_status": "reviewed", "reviewer": "SOFTWARE TEST FIXTURE ONLY",
                                      "reviewed_on": "2026-01-01", "rubric_id": self.rubric["rubric_id"],
                                      "rubric_version": self.rubric["version"]})
        self.write_manifest()

    def tearDown(self):
        self.temp.cleanup()

    def write_manifest(self):
        with self.manifest.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(self.rows)

    def test_rubric_refusal_and_order(self):
        pending = json.loads((ROOT / "severity-rubric-draft.json").read_text())
        with self.assertRaisesRegex(ValueError, "not approved"):
            validate_dataset(self.manifest, pending)
        invalid = copy.deepcopy(self.rubric); invalid["approved_by"] = ""
        with self.assertRaises(ValueError): validate_rubric(invalid)
        invalid = copy.deepcopy(self.rubric); invalid["labels"][1]["order"] = 3
        with self.assertRaisesRegex(ValueError, "contiguous"): validate_rubric(invalid)
        self.assertEqual(validate_rubric(self.rubric), ["none", "low", "moderate", "high"])

    def test_unreviewed_hash_and_site_refusal(self):
        for field, bad, message in (("review_status", "predicted", "reviewed"),
                                     ("source_sha256", "0" * 64, "SHA-256"),
                                     ("site", "Unknown", "Erbil/Amedi")):
            old = self.rows[0][field]; self.rows[0][field] = bad; self.write_manifest()
            with self.assertRaisesRegex(ValueError, message): validate_dataset(self.manifest, self.rubric)
            self.rows[0][field] = old

    def test_building_and_duplicate_leakage(self):
        for field in ("building_group", "duplicate_group"):
            old = self.rows[8][field]; self.rows[8][field] = self.rows[0][field]; self.write_manifest()
            with self.assertRaisesRegex(ValueError, "leakage"): validate_dataset(self.manifest, self.rubric)
            self.rows[8][field] = old
        self.rows[1]["image_path"] = self.rows[0]["image_path"]
        self.rows[1]["source_sha256"] = self.rows[0]["source_sha256"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate source"): validate_dataset(self.manifest, self.rubric)

    def test_missing_split_rejected(self):
        self.rows = [row for row in self.rows if row["split"] != "test"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "train, validation and test"): validate_dataset(self.manifest, self.rubric)

    def test_group_city_consistency_placeholders_and_site_coverage(self):
        for field in ("building_group", "duplicate_group"):
            old = self.rows[4][field]; self.rows[4][field] = self.rows[0][field]; self.write_manifest()
            with self.assertRaisesRegex(ValueError, "different cities"): validate_dataset(self.manifest, self.rubric)
            self.rows[4][field] = old
        old = self.rows[0]["building_group"]
        self.rows[0]["building_group"] = "unassigned"; self.write_manifest()
        with self.assertRaisesRegex(ValueError, "placeholder"): validate_dataset(self.manifest, self.rubric)
        self.rows[0]["building_group"] = old
        self.rows = [row for row in self.rows if not (row["site"] == "Amedi" and row["split"] == "test")]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "each split"): validate_dataset(self.manifest, self.rubric)

    def test_metrics_ordinal_error_absent_classes_and_empty_site(self):
        names = ["none", "low", "moderate", "high"]
        result = classification_metrics([0, 1, 3], [0, 3, 1], names)
        self.assertAlmostEqual(result["ordinal_mae"], 4 / 3)
        self.assertEqual(result["per_class"][2]["f1"], None)
        self.assertEqual(result["per_class"][2]["recall"], None)
        self.assertEqual(result["macro_f1_defined_classes"], 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        empty = classification_metrics([], [], names)
        self.assertIsNone(empty["macro_f1"]); self.assertIsNone(empty["ordinal_mae"])
        self.assertIsNone(empty["accuracy"])
        with self.assertRaises(ValueError): classification_metrics([.5], [0], names)

    def test_config_bounds_and_unavailable_model(self):
        invalid = dict(self.config, image_size=100)
        with self.assertRaises(ValueError): validate_config(invalid)
        with self.assertRaisesRegex(ValueError, "No severity model"):
            predict(self.root / "missing.pt", self.root / self.rows[0]["image_path"], self.root / "prediction")

    def test_unverified_encoder_weights_rejected_before_output(self):
        # Actual random tensor weights with synthetic target provenance: software
        # fixture only, representing an initializer whose source overlap is unknown.
        path = self.root / "unverified-target-encoder.pt"
        torch.save({"state_dict": SeverityCNN(4).state_dict(),
                    "training_sources": [self.rows[8]]}, path)
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(ValueError, "initialization is unavailable"):
            train(self.manifest, self.rubric_path, self.root / "forbidden-run", self.config, encoder_checkpoint=path)
        self.assertFalse((self.root / "forbidden-run").exists())

    def test_seed_checkpoint_inference_and_training_smoke(self):
        # Actual tiny CNN training checks the executable path; synthetic scores
        # have no relationship to heritage-damage performance and are discarded.
        seed_everything(17); first = SeverityCNN(4).state_dict()["head.weight"].clone()
        seed_everything(17); second = SeverityCNN(4).state_dict()["head.weight"].clone()
        self.assertTrue(torch.equal(first, second))
        results = train(self.manifest, self.rubric_path, self.root / "run", self.config)
        self.assertIsInstance(results["test"], str)
        self.assertIn("NOT EVALUATED", results["test"])
        checkpoint = self.root / "run/severity-best.pt"
        model, saved = load_checkpoint(checkpoint)
        self.assertEqual(saved["class_order"], ["none", "low", "moderate", "high"])
        path = self.root / self.rows[-1]["image_path"]
        a = predict(checkpoint, path, self.root / "predict-a")
        b = predict(checkpoint, path, self.root / "predict-b")
        self.assertEqual(a["raw_uncalibrated_probabilities"], b["raw_uncalibrated_probabilities"])
        self.assertAlmostEqual(sum(a["raw_uncalibrated_probabilities"].values()), 1., places=6)
        with self.assertRaisesRegex(ValueError, "already exists"):
            predict(checkpoint, path, self.root / "predict-a")
        with self.assertRaisesRegex(ValueError, "off by default"):
            evaluate(checkpoint, self.manifest, self.rubric_path, self.root / "test", split="test")
        report = evaluate(checkpoint, self.manifest, self.rubric_path, self.root / "evaluation")
        self.assertEqual(report["overall"]["count"], 8)
        self.assertEqual(report["sites"]["Erbil"]["count"], 4)
        self.assertEqual(report["sites"]["Amedi"]["count"], 4)
        test_report = evaluate(checkpoint, self.manifest, self.rubric_path, self.root / "test-allowed", split="test", allow_test=True)
        self.assertEqual(test_report["overall"]["count"], 8)
        # Original validation photos selected the epoch and cannot become a test
        # set merely by relabeling a different valid manifest.
        original = copy.deepcopy(self.rows)
        for row in self.rows:
            if row["split"] in ("validation", "test"):
                row["split"] = "test" if row["split"] == "validation" else "validation"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "model-selection validation"):
            evaluate(checkpoint, self.manifest, self.rubric_path, self.root / "selection-leak", split="test", allow_test=True)
        self.rows = original; self.write_manifest()
        legacy = dict(saved); legacy.pop("validation_sources")
        torch.save(legacy, self.root / "legacy.pt")
        with self.assertRaisesRegex(ValueError, "lacks original validation"):
            evaluate(self.root / "legacy.pt", self.manifest, self.rubric_path, self.root / "legacy-test", split="test", allow_test=True)
        # A different valid manifest cannot relabel trained photos as validation.
        for row in self.rows:
            if row["split"] in ("train", "validation"):
                row["split"] = "validation" if row["split"] == "train" else "train"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "overlaps checkpoint training"):
            evaluate(checkpoint, self.manifest, self.rubric_path, self.root / "leaked-evaluation")
        # Model metadata cannot silently reorder grades at inference.
        saved["class_order"] = list(reversed(saved["class_order"]))
        torch.save(saved, self.root / "tampered.pt")
        with self.assertRaisesRegex(ValueError, "class order"):
            load_checkpoint(self.root / "tampered.pt")


if __name__ == "__main__":
    unittest.main()
