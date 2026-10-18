# Training with labels approved through region 333

Completed a binary crack/intact pilot using 177,493 reviewed pixels from three images. The other pixels were ignored. Regions 334–336 were excluded. These are partial user-reviewed annotations, not complete expert-approved four-class masks.

## Results

| Held-out image | Original model crack Dice | Adapted model crack Dice | Interpretation |
|---|---:|---:|---|
| 0650 | 36.8% | 85.1% | Measured only on selected reviewed pixels. |
| 0450 | N/A | 0.0% | No positive labels in this image; adaptation produced 12,455 false-positive pixels. |
| 0750 | 21.6% | 59.3% | Measured only on selected reviewed pixels. |

Pooled crack Dice: **30.0% → 45.7%**. Pooled crack IoU: **29.6%**. Precision: **36.4%**; recall: **61.3%**.

Pixel accuracy is 90.0%, but predicting intact everywhere would score 93.1%. Pixel accuracy is misleading here because most reviewed pixels are intact. The 80% crack Dice target has not been reached. The 85.1% result on image0650 is not an overall score.

## Method

Started from the existing public-masonry U-Net/EfficientNet-B0 checkpoint. Froze the encoder and BatchNorm running statistics; trained the decoder with AdamW, learning rate0.0001, weight decay0.0001, masked BCE plus masked soft Dice. Used native-resolution128×128 patches, batch size4, two epochs of32steps each, paired rotations/flips, ImageNet normalization and seed333. Sampling balances available image/class combinations. All unreviewed255 pixels are excluded from both losses. Threshold0.5 and training duration were fixed before evaluating results.

For each of three folds, the model trained on the other two images and evaluated only the withheld image. No patches from a withheld image were used in that fold. Evaluation used a fixed non-overlapping128-pixel grid, counting each reviewed pixel once. The original checkpoint was evaluated on exactly the same pixels. No threshold or epoch was selected using these holdouts.

After the three folds, trained a separate final model on all three images. Its weights are `local-results/reviewed-through-333/unet-reviewed-final.pt`. The cross-validation metrics belong to the fold models, NOT to an independent evaluation of the final model. Keep the original model as well; the adapted model is an experimental candidate due to false positives.

## Limitations and next step

Only three source photos are represented; multiple regions from one photo are not independent examples. A later provenance audit confirmed all three photographs depict Erbil Citadel; images0450 and0650 also share a photographer. Building/scene overlap remains unverified. Reviewed pixels were chosen interactively rather than sampled randomly. Some dark lines may be joints, chipped edges or shadows, so annotation uncertainty remains. This is a same-site image-holdout diagnostic, not a new-site test, a four-class result, or structural safety assessment.

Image0450 has only intact labels. In its held-out fold, training had no intact examples from that photo, and false positives were substantial. Prioritize verified intact/joint/shadow examples from additional independent photos, plus complete masks on a small untouched test set. Improve label diversity and consistency before claiming broader accuracy.

## Files and running

- `train_reviewed_partial.py`: complete training/evaluation script.
- `local-results/reviewed-through-333/protocol.json`: source hashes, counts and fixed settings.
- `local-results/reviewed-through-333/results.json`: pooled/per-image metrics and training histories.
- `local-results/reviewed-through-333/*-mask.png`: frozen reviewed-label snapshots.
- `local-results/reviewed-through-333/*-out-of-fold.npz`: held-out probabilities; NaN outside evaluated tiles.
- `local-results/reviewed-through-333/verification.json`: completed output checks.
- `reviewed-training.log`: actual training log.

Run from the repository using `.venv-ml/bin/python train_reviewed_partial.py --output local-results/reviewed-through-333-repeat`. Each run requires a new output directory and existing source masks/checkpoint. Do not overwrite the recorded experiment. This script snapshots the current approved masks; compare protocol hashes to reproduce the exact inputs. The initial checkpoint and dependency versions must also match.

This historical run originally saved results and weights locally. See docs/ASSETS.md for the later distribution of selected labels, evidence and the newer binary checkpoint.
