# Run the first supervised pilot

This experiment trains three real networks using published binary crack masks from https://zenodo.org/records/18458458 (CC BY 4.0). It does not use the failed CLIPSeg pseudo masks or the 800 unlabeled Kurdish photos.

Open `Heritage.code-workspace` in VS Code, then run task **Train real UK binary pilot (not Kurdish results)**. Alternatively:

```sh
.venv-ml/bin/python train_binary_pilot.py
```

Requires the existing Python ML environment (`requirements-ml-lock.txt`) and `data/raw/masonry_crack_images.zip` / `masonry_crack_masks.zip`. The script verifies the published archive checksums against `audit_pilot.py`. It also reads `pilot-audit.json` for paired filenames. Run `audit_pilot.py` first if rebuilding from scratch.

Settings: CPU, two threads, EfficientNet-B0, random initialization (no encoder download), 256x256 fixed center crops, batch 2, Adam 0.001, binary focal + Dice, three epochs each for U-Net/FPN/DeepLabV3+, threshold 0.5. No CLAHE, augmentation or learning-rate warmup in this bounded diagnostic. It is a training workflow test, not the final instructor-specified recipe. Each model gets the same image split, but shuffled batch order differs as the DataLoader generator advances.

80 images train and 20 validate, split by seed 42 before cropping. Each source contributes one fixed center crop selected independently of the mask. Building identities are unknown, so scene leakage cannot be ruled out. These are development-validation scores, not an independent test, and these public images were previously used in our zero-shot CLIPSeg evaluation. Do not compare the two experiments directly: tasks, resolutions and fitting differ. This split is not the template for final Kurdish evaluation.

Results and exact split are saved under `local-results/binary-pilot/`. Checkpoints are diagnostic weights, not validated inspection tools. Full-size images and thin cracks outside center crops are not evaluated. Do not treat high background-dominated pixel accuracy as success.

## Target dataset preparation

See `ANNOTATION_PILOT.md`, `annotation-queue.csv` and local preview contacts. No four-class target masks are complete. After actual annotation and review, fill `training-manifest-template.csv`, preserving source hashes and assigning building groups before train/test splits. Validate with:

```sh
.venv-ml/bin/python validate_training_manifest.py your-manifest.csv
.venv-ml/bin/python -m unittest test_training_manifest -v
```

The validator checks declared review status, valid labels, dimensions, missing classes and group leakage. It cannot verify the honesty of a review declaration or the correctness of boundaries; a real reviewer must do that. It intentionally rejects the empty template.

## Revised U-Net experiment

The first three-epoch random-initialization runs predicted almost entirely background. A second development experiment uses an ImageNet-pretrained encoder, ImageNet RGB mean/std normalization, and weighted replacement sampling to balance crops with/without crack pixels in training only. It changes several factors at once, so it is not an ablation isolating one cause.

```sh
.venv-ml/bin/python train_binary_pilot.py --pretrained --balanced --epochs 5 --unet-only --output local-results/binary-balanced
```

The first pretrained attempt was stopped before a completed epoch to correct the missing ImageNet normalization; the documented revised run starts fresh. The validation set and threshold remain fixed. The comparison remains development-only; no untouched test set is available in this pilot. The model-library download uses its configured `smp-hub/efficientnet-b0.imagenet` source; record the resolved cache revision for exact provenance. All generated masks remain predictions, not approved ground truth.

Resolved pretrained encoder revision for this run: `1bbe7ecc1d5ea1d2058de1a2db063b8701aff314`; checksum in `pretrained-encoder-provenance.json`.

To reproduce the single Kurdish transfer illustration after the revised run:

```sh
.venv-ml/bin/python predict_pilot_sample.py
```

This predicts on a maximum-1024px preview of 0650 using overlapping 256px tiles. Red overlay is an unreviewed crack proposal. Non-predicted pixels are unknown (255), not healthy masonry. The preview mask is intentionally not a valid original-resolution four-class training mask.
