"""Native-resolution four-class inference; outputs are predictions, never severity.

Requires a local checkpoint/protocol from train_four_class.py. No encoder weights
are downloaded. Missing or incompatible models are errors, not synthetic grades.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from train_four_class import CLASSES, MODELS, starts, tensor

DISPLAY_CLASSES = ["Healthy masonry", "Visible crack", "Surface erosion", "Vegetation overgrowth"]
COLORS = np.array([[66, 160, 90], [235, 45, 65], [244, 165, 45], [150, 75, 205]], dtype=np.uint8)
MAX_PIXELS = 25_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(value):
    if "://" in str(value):
        raise ValueError("Only local files are supported; remote URLs are not loaded.")
    return Path(value).expanduser().resolve()


def validate_protocol(protocol, architecture):
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object.")
    if architecture not in MODELS:
        raise ValueError("Unsupported architecture.")
    declared = protocol.get("models")
    if not isinstance(declared, list) or not declared or not all(isinstance(n, str) and n in MODELS for n in declared):
        raise ValueError("Protocol must list supported training architectures in models.")
    if architecture not in declared:
        raise ValueError("Requested architecture is absent from the training protocol.")
    classes = protocol.get("classes")
    accepted = [CLASSES, DISPLAY_CLASSES]
    if classes not in accepted:
        raise ValueError("Protocol must declare the four expected classes in their training order.")
    for field in ("num_classes", "class_count"):
        if field in protocol and (type(protocol[field]) is not int or protocol[field] != 4):
            raise ValueError("Protocol class count must be four.")
    patch = protocol.get("patch")
    if type(patch) is not int or not 32 <= patch <= 1024 or patch % 32:
        raise ValueError("Protocol patch must be a multiple of 32 between 32 and 1024.")
    if type(protocol.get("no_clahe")) is not bool:
        raise ValueError("Protocol must explicitly declare boolean no_clahe preprocessing.")
    for field in ("encoder", "encoder_name"):
        if field in protocol and protocol[field] != "efficientnet-b0":
            raise ValueError("This trainer uses only the EfficientNet-B0 encoder.")
    if "in_channels" in protocol and (type(protocol["in_channels"]) is not int or protocol["in_channels"] != 3):
        raise ValueError("Protocol input must have three RGB channels.")
    if "normalization" in protocol and protocol["normalization"] != "ImageNet mean/std":
        raise ValueError("Unsupported normalization; training uses ImageNet mean/std.")
    if "clahe" in protocol and protocol["clahe"] is not (not protocol["no_clahe"]):
        raise ValueError("Conflicting CLAHE preprocessing declarations.")
    return patch, not protocol["no_clahe"]


def load_protocol(path, architecture):
    path = local_path(path)
    if not path.is_file() or path.stat().st_size > 1_000_000:
        raise ValueError("Protocol must be an existing JSON file smaller than 1 MB.")
    try:
        protocol = json.loads(path.read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Protocol is not valid UTF-8 JSON.") from error
    validate_protocol(protocol, architecture)
    return protocol


def load_model(checkpoint_path, protocol, architecture):
    validate_protocol(protocol, architecture)
    path = local_path(checkpoint_path)
    if not path.is_file():
        raise ValueError("Checkpoint is unavailable; supply an existing trained local state_dict.")
    # weights_only restricts deserialization. No fallback to unrestricted pickle.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("Checkpoint could not be loaded as a weights-only state_dict.") from error
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Checkpoint must contain a plain tensor state_dict.")
    if not all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point() or v.is_complex()):
        raise ValueError("Checkpoint contains nonfinite weights.")
    model = MODELS[architecture](encoder_name="efficientnet-b0", encoder_weights=None, classes=4)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Checkpoint architecture or class dimensions do not match the protocol/request.") from error
    return model.cpu().eval()


def verify_checkpoint_protocol_association(checkpoint_path, protocol_path, architecture):
    """Check new trainer sidecars; legacy weights retain an explicit warning."""
    checkpoint_path, protocol_path = map(local_path, (checkpoint_path, protocol_path))
    sidecar = checkpoint_path.with_suffix(".metadata.json")
    if not sidecar.exists():
        return {
            "status": "association unverified", "sidecar": None,
            "warning": "Legacy checkpoint has no association sidecar; architecture compatibility alone does not verify matching preprocessing.",
        }
    if not sidecar.is_file() or sidecar.stat().st_size > 1_000_000:
        raise ValueError("Checkpoint association sidecar must be a JSON file smaller than 1 MB.")
    try:
        association = json.loads(sidecar.read_text())
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Checkpoint association sidecar is not valid UTF-8 JSON.") from error
    if not isinstance(association, dict):
        raise ValueError("Checkpoint association sidecar must be a JSON object.")
    if association.get("architecture") != architecture or type(association.get("num_classes")) is not int or association["num_classes"] != 4:
        raise ValueError("Checkpoint association architecture or class count mismatch.")
    if not isinstance(association.get("scope"), str) or not association["scope"].strip():
        raise ValueError("Checkpoint association sidecar must declare its training scope.")
    for field, path in (("checkpoint_sha256", checkpoint_path), ("protocol_sha256", protocol_path)):
        expected = association.get(field)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Checkpoint association sidecar has an invalid {field}.")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Checkpoint association {field} mismatch; use the exact matching checkpoint and protocol.")
    return {
        "status": "association verified by matching hashes", "sidecar": str(sidecar),
        "sidecar_sha256": sha256(sidecar), "training_scope": association["scope"],
        "qualification": "Matching files establish artifact association, not human-label validity or model accuracy.",
    }


def validate_image(image):
    if not isinstance(image, Image.Image) or image.mode != "RGB":
        raise ValueError("Image must be an 8-bit RGB image.")
    if image.width < 1 or image.height < 1 or image.width * image.height > MAX_PIXELS:
        raise ValueError(f"Image must contain 1 to {MAX_PIXELS:,} pixels.")


def load_image(path):
    with Image.open(local_path(path)) as source:
        validate_image(source)
        return source.copy()


def validate_roi(roi, shape):
    if not isinstance(roi, np.ndarray) or roi.shape != shape or roi.ndim != 2:
        raise ValueError("ROI must be a two-dimensional mask matching the image dimensions exactly.")
    if roi.dtype == bool:
        valid = roi.copy()
    elif roi.dtype == np.uint8:
        values = set(np.unique(roi).tolist())
        if not (values <= {0, 1} or values <= {0, 255}):
            raise ValueError("ROI must be binary 0/1 or 0/255; mixed encodings are rejected.")
        valid = roi > 0
    else:
        raise ValueError("ROI must be boolean or uint8 binary data.")
    if not valid.any():
        raise ValueError("ROI is empty; at least one pixel must be included.")
    return valid


def load_roi(path, shape):
    with Image.open(local_path(path)) as source:
        if source.mode not in ("1", "L"):
            raise ValueError("ROI image must be a single-channel binary mask.")
        return validate_roi(np.array(source), shape)


@torch.inference_mode()
def predict_logits(model, image, patch, clahe):
    """Match train_four_class.evaluate: native tiles, PIL padding, mean logits."""
    validate_image(image)
    if type(patch) is not int or not 32 <= patch <= 1024 or patch % 32 or type(clahe) is not bool:
        raise ValueError("Invalid patch or CLAHE setting.")
    width, height = image.size
    scores = np.zeros((4, height, width), np.float32)
    counts = np.zeros((height, width), np.float32)
    model.cpu().eval()
    for top in starts(height, patch):
        for left in starts(width, patch):
            crop = image.crop((left, top, left + patch, top + patch))
            output = model(tensor(crop, clahe)[None])
            if not isinstance(output, torch.Tensor) or tuple(output.shape) != (1, 4, patch, patch):
                raise ValueError("Model returned incompatible four-class tile dimensions.")
            if not torch.isfinite(output).all():
                raise ValueError("Model returned nonfinite logits.")
            logits = output[0].cpu().numpy().astype(np.float32, copy=False)
            h, w = min(patch, height-top), min(patch, width-left)
            scores[:, top:top+h, left:left+w] += logits[:, :h, :w]
            counts[top:top+h, left:left+w] += 1
    if not np.all(counts > 0):
        raise ValueError("Internal tiling error left uncovered pixels.")
    return scores / counts[None]


def prediction_arrays(logits, roi=None):
    if not isinstance(logits, np.ndarray) or logits.ndim != 3 or logits.shape[0] != 4 or min(logits.shape[1:]) < 1 or not np.isfinite(logits).all():
        raise ValueError("Expected finite logits with shape 4 x H x W and nonempty dimensions.")
    valid = np.ones(logits.shape[1:], bool) if roi is None else validate_roi(roi, logits.shape[1:])
    mask = logits.argmax(axis=0).astype(np.uint8)
    probability = np.exp(logits - logits.max(axis=0, keepdims=True)).astype(np.float32)
    probability /= probability.sum(axis=0, keepdims=True)
    mask[~valid] = 255
    probability[:, ~valid] = np.nan
    denominator = int(valid.sum())
    summary = [{"id": i, "label": label, "pixels": int((mask == i).sum()),
                "fraction": float((mask == i).sum()/denominator)} for i, label in enumerate(DISPLAY_CLASSES)]
    return mask, probability, summary, valid


def overlay_image(image, mask, valid):
    original = np.array(image)
    overlay = original.copy()
    overlay[valid] = np.rint(.55*original[valid] + .45*COLORS[mask[valid]]).astype(np.uint8)
    return Image.fromarray(overlay)


def run(checkpoint, protocol_path, architecture, image_path, output_path, roi_path=None):
    checkpoint, protocol_path, image_path, output_path = map(local_path, (checkpoint, protocol_path, image_path, output_path))
    if output_path.exists():
        raise ValueError("Output already exists; choose a fresh directory to preserve previous results.")
    protocol = load_protocol(protocol_path, architecture)
    patch, clahe = validate_protocol(protocol, architecture)
    image = load_image(image_path)
    roi_path = None if roi_path is None else local_path(roi_path)
    roi = None if roi_path is None else load_roi(roi_path, (image.height, image.width))
    association = verify_checkpoint_protocol_association(checkpoint, protocol_path, architecture)
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    model = load_model(checkpoint, protocol, architecture)
    logits = predict_logits(model, image, patch, clahe)
    mask, probabilities, summary, valid = prediction_arrays(logits, roi)
    overlay = overlay_image(image, mask, valid)
    metadata = {
        "status": "UNVERIFIED MODEL PREDICTIONS; NOT GROUND TRUTH",
        "architecture": architecture, "encoder": "efficientnet-b0", "num_classes": 4,
        "class_labels": DISPLAY_CLASSES, "training_protocol_class_labels": protocol["classes"],
        "class_colors_rgb": COLORS.tolist(), "mask_encoding": "0..3 in class order; 255 outside supplied ROI",
        "probabilities": "Uncalibrated softmax of averaged logits, float32 shape [4,H,W]; NaN outside supplied ROI",
        "decision": "argmax of averaged logits; no severity classification or confidence calibration",
        "pixel_summary_scope": "Supplied ROI only; ROI suitability is not verified" if roi is not None else "Whole image, including background; not a masonry-only or wall-damage measurement",
        "measurement_warning": "Pixel fractions are descriptive predictions, not physical area, structural safety or damage severity.",
        "roi_supplied": roi is not None, "included_pixels": int(valid.sum()),
        "excluded_pixels": int((~valid).sum()), "class_summary": summary,
        "width": image.width, "height": image.height, "patch": patch,
        "preprocessing": {"clahe": clahe, "normalization": "ImageNet mean/std",
                          "tiling": "Training starts() grid, native resolution, PIL black padding for small images, average overlapping logits",
                          "orientation": "Stored pixel orientation; no automatic EXIF rotation, matching training"},
        "image": str(image_path), "image_sha256": sha256(image_path),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_protocol_association": association,
        "protocol": str(protocol_path), "protocol_sha256": sha256(protocol_path),
        "roi": None if roi_path is None else str(roi_path),
        "roi_sha256": None if roi_path is None else sha256(roi_path),
        "inference_script_sha256": sha256(Path(__file__)),
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "overlay": "prediction-overlay.png; 45% class color on source pixels, original size; outside ROI unchanged",
        "severity": "NOT ASSIGNED; no approved severity rubric or validated severity model is used",
        "accuracy": "NOT MEASURED; this inference command does not evaluate labeled ground truth",
    }
    # Atomic directory creation prevents replacing prior files, including races.
    output_path.mkdir(parents=True, exist_ok=False)
    Image.fromarray(mask).save(output_path / "prediction-classes.png")
    np.save(output_path / "prediction-probabilities.npy", probabilities, allow_pickle=False)
    overlay.save(output_path / "prediction-overlay.png")
    (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local trained four-class state_dict")
    parser.add_argument("--protocol", required=True, help="Matching training protocol.json")
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--image", required=True, help="Local 8-bit RGB image")
    parser.add_argument("--roi", help="Optional matching single-channel binary 0/1 or 0/255 masonry ROI")
    parser.add_argument("--output", required=True, help="Fresh output directory")
    args = parser.parse_args(argv)
    try:
        run(args.checkpoint, args.protocol, args.model, args.image, args.output, args.roi)
    except (ValueError, OSError, UnidentifiedImageError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Saved four-class model predictions, not severity or ground truth, to {args.output}")


if __name__ == "__main__":
    main()
