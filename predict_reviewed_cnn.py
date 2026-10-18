"""Local crack predictions using the reviewed CNN's exact evaluation tiling.

Outputs are model predictions, never human labels or ground truth. The checkpoint
must be a local U-Net EfficientNet-B0 single-output state dictionary; no weights
or images are downloaded. Image orientation is decoded as stored, matching the
context evaluator (no EXIF rotation or image resizing).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp

MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]
MAX_IMAGE_PIXELS = 64_000_000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("Checkpoint must be an existing local file.")
    model = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, classes=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def validate_tile_size(tile_size):
    if not isinstance(tile_size, int) or isinstance(tile_size, bool) or not 32 <= tile_size <= 2048 or tile_size % 32:
        raise ValueError("Tile size must be a multiple of 32 from 32 to 2048.")


def predict_probability(model, image, tile_size=512, device="cpu"):
    """Count each native image pixel once, with bottom/right edge padding only."""
    validate_tile_size(tile_size)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError("Image must be a nonempty HxWx3 uint8 RGB array.")
    height, width = image.shape[:2]
    probability = np.empty((height, width), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for top in range(0, height, tile_size):
            for left in range(0, width, tile_size):
                crop = image[top:top + tile_size, left:left + tile_size]
                h, w = crop.shape[:2]
                padded = np.pad(crop, ((0, tile_size - h), (0, tile_size - w), (0, 0)), mode="edge")
                tensor = (torch.from_numpy(padded.copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD
                logits = model(tensor[None].to(device))
                if tuple(logits.shape) != (1, 1, tile_size, tile_size):
                    raise ValueError("Checkpoint output must have shape (1, 1, tile_size, tile_size).")
                pred = logits.sigmoid()[0, 0, :h, :w].cpu().numpy()
                if not np.isfinite(pred).all():
                    raise ValueError("Model returned nonfinite predictions.")
                probability[top:top + h, left:left + w] = pred
    return probability


def predict_file(checkpoint, image_path, output, tile_size=512, threshold=.5, device="auto"):
    """Write a new prediction folder without overwriting any existing path."""
    validate_tile_size(tile_size)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite number from 0 to 1.")
    checkpoint, image_path, output = Path(checkpoint), Path(image_path), Path(output)
    if not checkpoint.is_file() or not image_path.is_file():
        raise ValueError("Checkpoint and image must both be existing local files.")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output path already exists; choose a new folder.")
    if device not in ("auto", "cpu", "mps"):
        raise ValueError("Device must be auto, cpu or mps.")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this machine; use cpu.")
    with Image.open(image_path) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("Image exceeds the 64 million pixel inference limit.")
        image = np.array(source.convert("RGB"))
    model = load_model(checkpoint, device)
    output.mkdir(parents=True, exist_ok=False)
    probability = predict_probability(model, image, tile_size=tile_size, device=device)
    mask = ((probability >= threshold).astype(np.uint8) * 255)
    probability_path, mask_path = output / "probability.npz", output / "predicted-crack-mask.png"
    np.savez_compressed(probability_path, probability=probability)
    Image.fromarray(mask).save(mask_path)
    height, width = probability.shape
    metadata = {
        "artifact_type": "MODEL PREDICTION ONLY; not ground truth or an approved annotation",
        "architecture": "U-Net EfficientNet-B0; single binary crack output",
        "image": str(image_path.resolve()), "image_sha256": sha(image_path),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha(checkpoint),
        "image_height": height, "image_width": width,
        "tile_size": tile_size, "threshold": threshold, "device": device,
        "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]},
        "inference": "Native non-overlapping top-left grid; bottom/right edge padding; no resizing, EXIF rotation, overlap, threshold tuning or TTA",
        "tile_count": ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size),
        "mask_values": {"0": "predicted non-crack", "255": "predicted crack; not the training ignore label"},
        "probability_file": probability_path.name, "probability_sha256": sha(probability_path),
        "mask_file": mask_path.name, "mask_sha256": sha(mask_path),
        "prediction_scope": "Binary visual crack candidate; not structural safety, four-class damage or physical severity",
        "evaluation": "No ground truth supplied; no accuracy, Dice or IoU measured",
        "script_sha256": sha(Path(__file__)),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        metadata = predict_file(args.checkpoint, args.image, args.output,
                                tile_size=args.tile_size, threshold=args.threshold, device=args.device)
    except (ValueError, FileExistsError, OSError, RuntimeError) as error:
        parser.exit(2, f"Prediction failed: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "shape": [metadata["image_height"], metadata["image_width"]],
                      "tile_count": metadata["tile_count"], "status": "prediction only; no accuracy measured"}))


if __name__ == "__main__":
    main()
