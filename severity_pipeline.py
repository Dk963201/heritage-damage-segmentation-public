"""Offline visual-severity CNN workflow; approved rubric and reviewed labels required.

No severity is inferred from damage-area cutoffs. Outputs describe visible image
appearance, not structural safety. This program never downloads model weights.
"""
import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

SITES = ("Erbil", "Amedi")
SPLITS = ("train", "validation", "test")
FIELDS = ("image_id", "image_path", "source_sha256", "site", "building_group",
          "duplicate_group", "split", "severity_label", "review_status", "reviewer",
          "reviewed_on", "rubric_id", "rubric_version")
PLACEHOLDERS = {"", "unknown", "pending", "unassigned", "tbd", "todo", "none", "n/a"}
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def named(value, field):
    if not isinstance(value, str) or value.strip().lower() in PLACEHOLDERS or not 1 <= len(value.strip()) <= 200:
        raise ValueError(f"{field} requires an actual recorded value, not a placeholder.")
    return value.strip()


def validate_rubric(rubric):
    if not isinstance(rubric, dict) or rubric.get("status") != "approved":
        raise ValueError("Severity rubric is not approved. Obtain a versioned rubric and named approver before training/evaluation.")
    for field in ("rubric_id", "version", "approved_by"):
        named(rubric.get(field), field)
    if "draft" in rubric["version"].lower():
        raise ValueError("Approved rubric needs a released version, not a draft version.")
    try:
        date.fromisoformat(rubric["approved_on"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Approved rubric requires approved_on as YYYY-MM-DD.") from None
    if rubric.get("scope") != "visible_damage_only" or rubric.get("ordered") is not True or rubric.get("structural_safety_inference_allowed") is not False:
        raise ValueError("Rubric must explicitly define ordered visible-damage grades and exclude structural safety inference.")
    labels = rubric.get("labels")
    if not isinstance(labels, list) or not 2 <= len(labels) <= 10:
        raise ValueError("Rubric needs 2–10 ordered labels.")
    ids = []
    for index, label in enumerate(labels):
        if not isinstance(label, dict) or type(label.get("order")) is not int or label["order"] != index:
            raise ValueError("Rubric labels must appear in contiguous order starting at zero.")
        ident = label.get("id", "")
        if not isinstance(ident, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", ident) or ident in ids:
            raise ValueError("Rubric label IDs must be unique lowercase identifiers.")
        named(label.get("definition"), "label definition")
        ids.append(ident)
    return ids


def read_rgb(path):
    with Image.open(path) as image:
        if image.width * image.height > 64_000_000:
            raise ValueError("Image exceeds the 64 million pixel limit.")
        return image.convert("RGB").copy()


def image_path(root, text):
    path = Path(text)
    if not text or path.is_absolute():
        raise ValueError("image_path must be relative to the manifest directory.")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"Missing image or path outside manifest directory: {text}")
    return resolved


def validate_dataset(manifest, rubric):
    classes = validate_rubric(rubric)
    manifest = Path(manifest).resolve()
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError("Severity manifest is missing required columns.")
        rows = [{key: (row.get(key) or "").strip() for key in FIELDS} for row in reader]
    if not rows or len(rows) > 10000:
        raise ValueError("Manifest must contain 1–10,000 reviewed image-level severity labels.")
    root = manifest.parent
    ids, hashes, pixel_hashes, buildings, duplicates = set(), set(), set(), {}, {}
    for row in rows:
        for field in ("image_id", "building_group", "duplicate_group", "reviewer"):
            named(row[field], field)
        if row["image_id"] in ids:
            raise ValueError("Duplicate image_id in manifest.")
        ids.add(row["image_id"])
        if row["review_status"] != "reviewed":
            raise ValueError("Every severity label must be independently reviewed; predictions/pending labels are prohibited.")
        if row["site"] not in SITES or row["split"] not in SPLITS:
            raise ValueError("Each image must use site Erbil/Amedi and split train/validation/test.")
        if row["severity_label"] not in classes:
            raise ValueError("Severity label is absent from the approved rubric.")
        if (row["rubric_id"], row["rubric_version"]) != (rubric["rubric_id"], rubric["version"]):
            raise ValueError("Label rubric ID/version differs from approved rubric.")
        try:
            date.fromisoformat(row["reviewed_on"])
        except ValueError:
            raise ValueError("Every reviewed label needs reviewed_on as YYYY-MM-DD.") from None
        path = image_path(root, row["image_path"])
        digest = sha(path)
        if not re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"]) or digest != row["source_sha256"]:
            raise ValueError("Source image SHA-256 mismatch or missing hash.")
        if digest in hashes:
            raise ValueError("Duplicate source image bytes are prohibited across the entire manifest.")
        hashes.add(digest)
        image = read_rgb(path)
        pixel_digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if pixel_digest in pixel_hashes:
            raise ValueError("Duplicate decoded RGB image is prohibited, including metadata-only variants.")
        pixel_hashes.add(pixel_digest)
        row["decoded_rgb_sha256"] = pixel_digest
        for field, seen in (("building_group", buildings), ("duplicate_group", duplicates)):
            group = row[field]
            if group in seen and seen[group][1] != row["split"]:
                raise ValueError(f"{field} leakage: group {group!r} spans multiple splits.")
            if group in seen and seen[group][0] != row["site"]:
                raise ValueError(f"{field} cannot identify the same group in different cities.")
            seen[group] = (row["site"], row["split"])
    if {row["split"] for row in rows} != set(SPLITS):
        raise ValueError("Nonempty train, validation and test splits are required.")
    for split in SPLITS:
        if {row["site"] for row in rows if row["split"] == split} != set(SITES):
            raise ValueError(f"Both Erbil and Amedi are required in each split; incomplete {split} coverage.")
    if {row["severity_label"] for row in rows if row["split"] == "train"} != set(classes):
        raise ValueError("Training split must contain reviewed examples of every rubric class.")
    return rows, root, classes


def classification_metrics(truth, predicted, classes):
    n = len(classes)
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if any(array.size and array.dtype.kind not in "iu" for array in (truth, predicted)):
        raise ValueError("Metric labels must be integer class indices.")
    truth, predicted = truth.astype(np.int64), predicted.astype(np.int64)
    if truth.shape != predicted.shape or truth.ndim != 1 or np.any(truth < 0) or np.any(truth >= n) or np.any(predicted < 0) or np.any(predicted >= n):
        raise ValueError("Metric labels must be equally sized vectors of valid class indices.")
    cm = np.bincount(truth * n + predicted, minlength=n * n).reshape(n, n)
    per_class = []
    for i, name in enumerate(classes):
        tp, actual, proposed = int(cm[i, i]), int(cm[i].sum()), int(cm[:, i].sum())
        per_class.append({"label": name, "support": actual, "predicted_count": proposed,
                          "precision": tp / proposed if proposed else None,
                          "recall": tp / actual if actual else None,
                          "f1": 2 * tp / (actual + proposed) if actual + proposed else None})
    defined = [row["f1"] for row in per_class if row["f1"] is not None]
    return {"class_order": classes, "count": len(truth), "confusion_matrix": cm.tolist(),
            "per_class": per_class, "macro_f1": float(np.mean(defined)) if defined else None,
            "macro_f1_defined_classes": len(defined),
            "absent_true_classes": [r["label"] for r in per_class if not r["support"]],
            "accuracy": float((truth == predicted).mean()) if len(truth) else None,
            "ordinal_mae": float(np.abs(truth - predicted).mean()) if len(truth) else None,
            "ordinal_error_unit": "rubric class steps; not physical damage units"}


def preprocess(image, size):
    image = ImageOps.contain(image.convert("RGB"), (size, size), method=Image.Resampling.BILINEAR)
    padded = Image.new("RGB", (size, size), (0, 0, 0))
    padded.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    x = torch.from_numpy(np.array(padded).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


class SeverityCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = smp.encoders.get_encoder("efficientnet-b0", weights=None, depth=5)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.head = torch.nn.Linear(self.encoder.out_channels[-1], num_classes)

    def forward(self, x):
        return self.head(self.pool(self.encoder(x)[-1]).flatten(1))


class SeverityImages(Dataset):
    def __init__(self, rows, root, classes, size):
        self.rows, self.root, self.classes, self.size = rows, root, classes, size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return preprocess(read_rgb(image_path(self.root, row["image_path"])), self.size), self.classes.index(row["severity_label"])


def validate_config(config):
    for key, low, high in (("seed", 0, 2**31 - 1), ("epochs", 1, 200), ("batch_size", 1, 32), ("image_size", 96, 512)):
        if type(config.get(key)) is not int or not low <= config[key] <= high:
            raise ValueError(f"Invalid bounded training parameter: {key}")
    if config["image_size"] % 32 or not isinstance(config.get("lr"), (int, float)) or not np.isfinite(config["lr"]) or not 1e-7 <= config["lr"] <= .01:
        raise ValueError("Image size must be a multiple of 32; learning rate must be 1e-7–0.01.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def score(model, rows, root, classes, size):
    model.eval()
    truth, predicted, records = [], [], []
    for row in rows:
        x = preprocess(read_rgb(image_path(root, row["image_path"])), size)
        probabilities = model(x[None]).softmax(1)[0].numpy()
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite severity probabilities.")
        actual, pred = classes.index(row["severity_label"]), int(probabilities.argmax())
        truth.append(actual); predicted.append(pred)
        records.append({"image_id": row["image_id"], "site": row["site"],
                        "truth": classes[actual], "predicted": classes[pred],
                        "raw_uncalibrated_probabilities": dict(zip(classes, map(float, probabilities)))})
    overall = classification_metrics(truth, predicted, classes)
    slices = {}
    for site in SITES:
        ix = [i for i, row in enumerate(rows) if row["site"] == site]
        slices[site] = classification_metrics([truth[i] for i in ix], [predicted[i] for i in ix], classes)
    return {"overall": overall, "sites": slices, "predictions": records}


def new_output(path):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("Output path already exists; use a new directory.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("No severity model is available at that path. An approved rubric and reviewed dataset must be trained first.")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format") != "heritage-severity-v1":
        raise ValueError("Checkpoint is not a supported severity model.")
    classes = validate_rubric(saved.get("rubric"))
    validate_config(saved.get("config", {}))
    if saved.get("class_order") != classes or saved.get("rubric_content_sha256") != object_sha(saved["rubric"]):
        raise ValueError("Checkpoint class order or rubric hash is inconsistent.")
    model = SeverityCNN(len(classes))
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def train(manifest, rubric_path, output, config, encoder_checkpoint=None, evaluate_test=False):
    if encoder_checkpoint is not None:
        raise ValueError("Custom encoder initialization is unavailable until its training-source provenance is verified against validation/test groups. Use random initialization.")
    validate_config(config)
    rubric = json.loads(Path(rubric_path).read_text())
    rows, root, classes = validate_dataset(manifest, rubric)
    seed_everything(config["seed"])
    out = new_output(output)
    splits = {name: [r for r in rows if r["split"] == name] for name in SPLITS}
    model = SeverityCNN(len(classes))
    protocol = {"config": config, "class_order": classes, "rubric": rubric,
                "rubric_file_sha256": sha(rubric_path), "rubric_content_sha256": object_sha(rubric),
                "manifest_sha256": sha(manifest), "dataset_content_sha256": object_sha(rows),
                "initialization": "random weights throughout; no downloads; custom initializer provenance unverified and therefore disabled",
                "selection": "Maximum validation macro F1; earliest epoch on ties; test excluded from selection",
                "test_requested": bool(evaluate_test), "architecture": "EfficientNet-B0 encoder + global average pooling + linear ordered-class head",
                "preprocessing": "RGB; aspect-preserving bilinear resize to fit square, centered black padding, ImageNet normalization; no augmentation",
                "optimizer": "AdamW, weight_decay=0.0001", "loss": "Cross entropy",
                "device": "cpu", "torch": str(torch.__version__), "smp": smp.__version__,
                "script_sha256": sha(Path(__file__)),
                "limitations": ["Visual appearance only; not structural safety", "Raw probabilities are not calibrated", "Near-duplicate and building identity rely on recorded human grouping"]}
    write_json(out / "protocol.json", protocol)
    write_json(out / "sources.json", rows)
    write_json(out / "rubric.json", rubric)
    loader = DataLoader(SeverityImages(splits["train"], root, classes, config["image_size"]),
                        batch_size=config["batch_size"], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    best, history, selected_epoch = -1., [], None
    for epoch in range(config["epochs"]):
        model.train()
        loss_total, seen = 0., 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_total += float(loss.detach()) * len(y); seen += len(y)
        validation = score(model, splits["validation"], root, classes, config["image_size"])
        value = validation["overall"]["macro_f1"]
        history.append({"epoch": epoch + 1, "training_loss": loss_total / seen, "validation": validation})
        if value is not None and value > best:
            best, selected_epoch = value, epoch + 1
            torch.save({"format": "heritage-severity-v1", "state_dict": model.state_dict(),
                        "class_order": classes, "rubric": rubric, "rubric_content_sha256": object_sha(rubric),
                        "config": config, "selected_epoch": selected_epoch,
                        "training_manifest_sha256": protocol["manifest_sha256"],
                        "training_data_sha256": protocol["dataset_content_sha256"],
                        "training_sources": [{k: r[k] for k in ("image_id", "source_sha256", "decoded_rgb_sha256", "building_group", "duplicate_group")} for r in splits["train"]],
                        "validation_sources": [{k: r[k] for k in ("image_id", "source_sha256", "decoded_rgb_sha256", "building_group", "duplicate_group")} for r in splits["validation"]]}, out / "severity-best.pt")
        write_json(out / "history.json", history)
        print(json.dumps({"epoch": epoch + 1, "validation_macro_f1": value}), flush=True)
    if selected_epoch is None:
        raise ValueError("No defined validation macro F1; no model selected.")
    model, _ = load_checkpoint(out / "severity-best.pt")
    results = {"selected_epoch": selected_epoch, "best_validation_macro_f1": best,
               "model_sha256": sha(out / "severity-best.pt"),
               "validation": score(model, splits["validation"], root, classes, config["image_size"]),
               "test": "NOT EVALUATED; opt in only after freezing model choices"}
    if evaluate_test:
        results["test"] = score(model, splits["test"], root, classes, config["image_size"])
    for row in rows:
        if sha(image_path(root, row["image_path"])) != row["source_sha256"]:
            raise ValueError("Source image changed during training.")
    write_json(out / "results.json", results)
    return results


def evaluate(checkpoint, manifest, rubric_path, output, split="validation", allow_test=False):
    if split not in ("validation", "test") or (split == "test" and not allow_test):
        raise ValueError("Test evaluation is off by default. Use --allow-test only after freezing model choices.")
    rubric = json.loads(Path(rubric_path).read_text())
    rows, root, classes = validate_dataset(manifest, rubric)
    model, saved = load_checkpoint(checkpoint)
    if classes != saved["class_order"] or object_sha(rubric) != saved["rubric_content_sha256"]:
        raise ValueError("Evaluation rubric differs from model rubric.")
    selected = [row for row in rows if row["split"] == split]
    excluded_sources = list(saved["training_sources"])
    if split == "test":
        if not isinstance(saved.get("validation_sources"), list) or not saved["validation_sources"]:
            raise ValueError("Checkpoint lacks original validation-source provenance; independent test evaluation is unavailable.")
        excluded_sources += saved["validation_sources"]
    # Also prevent a different manifest from putting the checkpoint's own training
    # images, building, or duplicate groups into an evaluation split.
    for row in selected:
        for trained in excluded_sources:
            if any(row[k] == trained[k] for k in ("image_id", "source_sha256", "decoded_rgb_sha256", "building_group", "duplicate_group")):
                raise ValueError("Evaluation overlaps checkpoint training or model-selection validation images/groups.")
    out = new_output(output)
    seed_everything(saved["config"]["seed"])
    result = score(model, selected, root, classes, saved["config"]["image_size"])
    result.update(split=split, model_sha256=sha(checkpoint), manifest_sha256=sha(manifest),
                  dataset_content_sha256=object_sha(rows), rubric_sha256=sha(rubric_path))
    write_json(out / "results.json", result)
    return result


def predict(checkpoint, image, output):
    model, saved = load_checkpoint(checkpoint)
    seed_everything(saved["config"]["seed"])
    image = Path(image)
    if not image.is_file():
        raise ValueError("Prediction image must be an existing local file.")
    x = preprocess(read_rgb(image), saved["config"]["image_size"])
    with torch.inference_mode():
        probs = model(x[None]).softmax(1)[0].numpy()
    if not np.isfinite(probs).all():
        raise ValueError("Nonfinite severity probabilities.")
    classes = saved["class_order"]
    result = {"status": "MODEL PREDICTION; not a reviewed label or structural safety assessment",
              "predicted_visual_severity": classes[int(probs.argmax())],
              "class_order": classes, "raw_uncalibrated_probabilities": dict(zip(classes, map(float, probs))),
              "rubric_id": saved["rubric"]["rubric_id"], "rubric_version": saved["rubric"]["version"],
              "rubric_content_sha256": saved["rubric_content_sha256"],
              "image_sha256": sha(image), "model_sha256": sha(checkpoint),
              "evaluation": "No reference label supplied; no accuracy measured"}
    write_json(new_output(output) / "prediction.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "train", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("manifest", type=Path)
        p.add_argument("--rubric", type=Path, required=True)
        if name != "validate":
            p.add_argument("--output", type=Path, required=True)
    p = sub.choices["train"]
    p.add_argument("--epochs", type=int, default=20); p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=333)
    p.add_argument("--random-init", action="store_true", help="Explicitly choose the offline random-initialization default")
    p.add_argument("--evaluate-test", action="store_true")
    p = sub.choices["evaluate"]
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--allow-test", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            rows, _, classes = validate_dataset(args.manifest, json.loads(args.rubric.read_text()))
            print(json.dumps({"status": "validated", "images": len(rows), "class_order": classes,
                              "split_counts": {s: sum(r["split"] == s for r in rows) for s in SPLITS}}))
        elif args.command == "train":
            config = {key: getattr(args, key) for key in ("seed", "epochs", "batch_size", "image_size", "lr")}
            train(args.manifest, args.rubric, args.output, config, evaluate_test=args.evaluate_test)
        elif args.command == "evaluate":
            evaluate(args.checkpoint, args.manifest, args.rubric, args.output, args.split, args.allow_test)
        else:
            print(json.dumps(predict(args.checkpoint, args.image, args.output)))
    except (ValueError, OSError, RuntimeError, KeyError, csv.Error) as error:
        parser.exit(2, f"Severity workflow unavailable: {error}\n")


if __name__ == "__main__":
    main()
