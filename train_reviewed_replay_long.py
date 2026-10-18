"""One final prespecified longer public-replay candidate on sparse reviewed labels.

This is development evaluation: the three target photographs were already viewed
and their earlier fold results informed this candidate. It is not a new test.
"""
import argparse
import hashlib
import io
import json
import platform
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import PIL
import torch
import segmentation_models_pytorch as smp

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "local-results/reviewed-through-333"
CHECKPOINT = ROOT / "local-results/public-masonry/unet-diagnostic.pt"
SPLIT = ROOT / "local-results/public-masonry/split.json"
SEED, PATCH, EPOCHS, STEPS = 333, 128, 12, 32
EVAL_TILE = 512
PREVIOUS_BEST_DICE = 0.685586
MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def metrics(cm):
    tn, fp, fn, tp = map(int, cm.ravel())
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "pixel_accuracy": float((tn + tp) / cm.sum()),
        "confusion_tn_fp_fn_tp": [tn, fp, fn, tp],
        "reviewed_pixels": int(cm.sum()),
    }


def tensor(array):
    return (torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255 - MEAN) / STD


def augment(x, y, rng):
    k = int(rng.integers(4))
    x, y = np.rot90(x, k), np.rot90(y, k)
    if rng.integers(2):
        x, y = x[:, ::-1], y[:, ::-1]
    return tensor(x), torch.from_numpy(y.copy())


def masked_loss(logits, labels, dice_weight):
    valid = labels != 255
    assert valid.any()
    target = (labels == 1).float()[valid]
    z = logits[valid]
    bce = torch.nn.functional.binary_cross_entropy_with_logits(z, target)
    prob = z.sigmoid()
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice_weight * dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "local-results/reviewed-replay-long-v1"))
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    previous_protocol = json.loads((FROZEN / "protocol.json").read_text())
    assert sha(CHECKPOINT) == previous_protocol["checkpoint_sha256"]
    data, inputs = {}, []
    for version in previous_protocol["versions"]:
        ident = version["id"]
        ip = ROOT / version["image"]
        mp = FROZEN / f"{ident}-mask.png"
        assert sha(ip) == version["image_sha256"]
        assert sha(mp) == version["mask_sha256"]
        x = np.array(Image.open(ip).convert("RGB"))
        y = np.array(Image.open(mp))
        assert y.shape == x.shape[:2] and set(np.unique(y)) <= {0, 1, 255}
        data[ident] = x, y
        shutil.copy2(mp, out / mp.name)
        inputs.append({**version, "frozen_mask": str(mp.relative_to(ROOT))})
    pending = ROOT / "data/processed/review-0750/batch-27"
    manifest = json.loads((pending / "review-manifest.json").read_text())
    assert manifest["accepted_region_ids"] == list(range(328, 334))
    assert manifest["pending_region_ids"] == [334, 335, 336]
    for ident in manifest["pending_region_ids"]:
        selected = np.array(Image.open(pending / f"region-{ident}-selection.png")) > 0
        assert np.all(data["0750"][1][selected] == 255)

    # Reuse and verify the exact cached public pool from the conservative run.
    cache = ROOT / "local-results/reviewed-replay-v1"
    cache_protocol = json.loads((cache / "protocol.json").read_text())
    assert sha(cache / "public-replay-pool.npz") == cache_protocol["public_pool_sha256"]
    assert sha(cache / "public-replay-crops.json") == cache_protocol["public_crop_manifest_sha256"]
    assert sha(SPLIT) == cache_protocol["public_split_sha256"]
    split = json.loads(SPLIT.read_text())
    train_names = split["train"]
    assert len(train_names) == 80 and not set(train_names) & set(split["validation"])
    assert train_names == cache_protocol["public_train"]
    archives = cache_protocol["archives"]
    for archive in archives:
        assert sha(ROOT / archive["path"]) == archive["sha256"]
    loaded = np.load(cache / "public-replay-pool.npz")
    public_x, public_y = loaded["images"], loaded["masks"]
    assert public_x.shape == (640, PATCH, PATCH, 3)
    assert public_y.shape == (640, PATCH, PATCH)
    assert public_x.dtype == public_y.dtype == np.uint8
    assert set(np.unique(public_y)) <= {0, 1}
    crop_records = json.loads((cache / "public-replay-crops.json").read_text())
    assert len(crop_records) == len(public_x)
    assert set(row["source"] for row in crop_records) == set(train_names)
    shutil.copy2(cache / "public-replay-pool.npz", out / "public-replay-pool.npz")
    shutil.copy2(cache / "public-replay-crops.json", out / "public-replay-crops.json")
    protocol = {
        "scope": "DEVELOPMENT image-holdout diagnostic; already-seen target photographs, not independent test",
        "prespecification": "One last fixed12epoch candidate; no intermediate held-out evaluations, no epoch or threshold selection",
        "seed": SEED, "device": str(device), "architecture": "U-Net EfficientNet-B0",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)), "checkpoint_sha256": sha(CHECKPOINT),
        "encoder": "frozen", "batchnorm": "frozen running statistics", "training_patch": PATCH, "evaluation_tile": EVAL_TILE,
        "batch_size": 4, "sparse_target_patches_per_batch": 2, "public_patches_per_batch": 2,
        "epochs": EPOCHS, "steps_per_epoch": STEPS, "optimizer": "AdamW",
        "lr": 2e-5, "weight_decay": 1e-3,
        "loss": "target masked BCE +0.25 masked soft Dice; plus public BCE +soft Dice; ignore255 excluded",
        "threshold": .5, "augmentation": "paired rotations/flips; ImageNet normalization",
        "target_sampling": "uniform available image/class combination then uniform labeled pixel; jitter +/-24",
        "public_sampling": "uniform crop from fixed pool:8 native128 crops per80 UK train photos,4uniform/4crack-centered",
        "public_pool_sha256": sha(out / "public-replay-pool.npz"),
        "public_crop_manifest_sha256": sha(out / "public-replay-crops.json"),
        "public_split_sha256": sha(SPLIT), "public_train": train_names,
        "public_validation_excluded": split["validation"], "archives": archives,
        "evaluation": "same3 leave-one-image-out folds; fixed non-overlapping512 native-resolution grid; each reviewed pixel counted once",
        "final_training_rule": "Train all3 only if pooled development Dice exceeds0.685586; do not assign fold metrics to final model",
        "pending_regions_excluded": [334, 335, 336], "target_inputs": inputs,
        "script_sha256": sha(Path(__file__)),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "smp": smp.__version__, "numpy": np.__version__, "Pillow": PIL.__version__},
        "limitations": ["Only3targetphotos with selected partial labels from the same Erbil Citadel site",
                        "Target holdouts already inspected; repeated experiments are development evaluation",
                        "Unknown crack boundaries and uncertain dark-line semantics; not four-class labels",
                        "Public/target scene near-duplicate overlap not comprehensively audited",
                        "No independent final-model test; conditional all-target training only after all folds"],
    }
    write_json(out / "protocol.json", protocol)

    def make_model():
        torch.manual_seed(SEED)
        m = smp.Unet(encoder_name="efficientnet-b0", encoder_weights=None, classes=1)
        m.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
        for param in m.encoder.parameters():
            param.requires_grad = False
        return m.to(device)

    def evaluate(m, ident):
        m.eval()
        x, y = data[ident]
        cm = np.zeros((2, 2), np.int64)
        prob = np.full(y.shape, np.nan, np.float32)
        with torch.inference_mode():
            for top in range(0, y.shape[0], EVAL_TILE):
                for left in range(0, y.shape[1], EVAL_TILE):
                    yy = y[top:top + EVAL_TILE, left:left + EVAL_TILE]
                    valid = yy != 255
                    if not valid.any():
                        continue
                    xx = x[top:top + EVAL_TILE, left:left + EVAL_TILE]
                    h, w = yy.shape
                    padded = np.pad(xx, ((0, EVAL_TILE-h), (0, EVAL_TILE-w), (0, 0)), mode="edge")
                    pp = m(tensor(padded)[None].to(device)).sigmoid()[0, 0, :h, :w].cpu().numpy()
                    prob[top:top+h, left:left+w] = pp
                    cm += np.bincount(yy[valid].astype(int)*2 + (pp[valid] >= .5), minlength=4).reshape(2, 2)
        assert int(cm.sum()) == int((y != 255).sum())
        return cm, prob

    def train(m, ids, tag):
        rng = np.random.default_rng(SEED)
        coords = {(i, c): np.argwhere(data[i][1] == c) for i in ids for c in [0, 1]}
        keys = [k for k, v in coords.items() if len(v)]
        optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5, weight_decay=1e-3)
        history, seen_public = [], set()
        for epoch in range(EPOCHS):
            m.train()
            for module in m.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            losses = []
            for step in range(STEPS):
                xx, yy = [], []
                for _ in range(2):
                    key = keys[int(rng.integers(len(keys)))]
                    i, _ = key
                    cy, cx = coords[key][int(rng.integers(len(coords[key])))]
                    x, y = data[i]
                    top = int(np.clip(cy-64+rng.integers(-24, 25), 0, y.shape[0]-PATCH))
                    left = int(np.clip(cx-64+rng.integers(-24, 25), 0, y.shape[1]-PATCH))
                    a, b = augment(x[top:top+PATCH, left:left+PATCH], y[top:top+PATCH, left:left+PATCH], rng)
                    xx.append(a)
                    yy.append(b)
                for _ in range(2):
                    j = int(rng.integers(len(public_x)))
                    seen_public.add(crop_records[j]["source"])
                    a, b = augment(public_x[j], public_y[j], rng)
                    xx.append(a)
                    yy.append(b)
                x = torch.stack(xx).to(device)
                y = torch.stack(yy).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = m(x)[:, 0]
                loss = masked_loss(logits[:2], y[:2], .25) + masked_loss(logits[2:], y[2:], 1.)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            rec = {"epoch": epoch+1, "loss": float(np.mean(losses))}
            history.append(rec)
            print(tag, rec, flush=True)
            write_json(out / f"{tag}-training-history.json", history)
        return history, sorted(seen_public)

    results = []
    total, baseline = np.zeros((2, 2), np.int64), np.zeros((2, 2), np.int64)
    for held in data:
        started = time.monotonic()
        m = make_model()
        before, _ = evaluate(m, held)
        ids = [i for i in data if i != held]
        history, seen_public = train(m, ids, "holdout-" + held)
        # Freeze the candidate weights/hash before evaluating this fold.
        cp = out / f"{held}-fold.pt"
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, cp)
        fold_meta = {"held_out": held, "train_images": ids,
                     "checkpoint": cp.name, "checkpoint_sha256": sha(cp),
                     "public_train_sources_seen": seen_public, "history": history}
        write_json(out / f"{held}-fold-protocol.json", fold_meta)
        after, probabilities = evaluate(m, held)
        baseline += before
        total += after
        np.savez_compressed(out / f"{held}-out-of-fold.npz", probability=probabilities)
        rec = {**fold_meta, "before": metrics(before), "after": metrics(after),
               "seconds": time.monotonic()-started}
        results.append(rec)
        write_json(out / "fold-results.json", results)
        print("RESULT", held, rec["after"], flush=True)
        del m
        if device.type == "mps":
            torch.mps.empty_cache()
    result = {"scope": protocol["scope"], "protocol_sha256": sha(out / "protocol.json"),
              "baseline_pooled": metrics(baseline), "adapted_pooled": metrics(total),
              "folds": results, "final_model": "Not trained; fold results only"}
    result["previous_best_development_dice"] = PREVIOUS_BEST_DICE
    result["beats_previous_best"] = result["adapted_pooled"]["dice"] > PREVIOUS_BEST_DICE
    write_json(out / "results.json", result)
    print("FOLDS COMPLETE", json.dumps(result["adapted_pooled"]), flush=True)
    if result["beats_previous_best"]:
        m = make_model()
        history, seen_public = train(m, list(data), "final-all-images")
        cp = out / "unet-reviewed-replay-long-final.pt"
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, cp)
        result["final_model"] = {
            "checkpoint": cp.name, "checkpoint_sha256": sha(cp),
            "training_images": list(data), "training_history": history,
            "public_train_sources_seen": seen_public,
            "evaluation": "Not independently evaluated. Fold scores belong to separate fold models.",
        }
        write_json(out / "results.json", result)
    print("COMPLETE", json.dumps({"metrics":result["adapted_pooled"], "beats_previous_best":result["beats_previous_best"]}), flush=True)


if __name__ == "__main__":
    main()
