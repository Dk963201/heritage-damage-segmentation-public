# Reviewed-label improvement experiments

Best pooled crack Dice: **69.2%**, compared with **45.7%** for the earlier target adaptation. This does not reach the 80% target. It is a reused development result, not independently verified accuracy.

Selected candidate: **CNN with public replay, 12 epochs, 512px**. Precision **95.9%**; recall **54.2%**; IoU **52.9%**.

## What was measured

All runs score the same 177,493 user-reviewed pixels: 165,275 intact and 12,218 crack, from images 0650, 0450 and 0750. Each target-trained fold excludes its evaluation photograph; the public baseline has no target training. Regions 334–336 and all unreviewed 255 pixels remain excluded. No label was changed to improve a score.

The metric is pooled crack Dice = 2TP / (2TP + FP + FN). Raw pixel accuracy is unsuitable as the main target: predicting intact everywhere already yields 93.1% on these selected pixels. No experiment below used a threshold search; all thresholds are 0.5.

## Complete comparison

| Candidate | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Original public CNN, 128px | 29.95% | 17.61% | 100.00% | 17.61% |
| First target adaptation, 128px | 45.69% | 29.61% | 36.41% | 61.30% |
| CNN with public replay, 4 epochs, 128px | 56.38% | 39.26% | 99.83% | 39.29% |
| Texture random_forest | 53.66% | 36.66% | 39.66% | 82.93% |
| Texture hist_gradient_boosting | 51.70% | 34.87% | 36.62% | 87.94% |
| Equal average: replay + random_forest | 50.61% | 33.88% | 48.69% | 52.69% |
| Equal average: replay + hist_gradient_boosting | 46.30% | 30.13% | 42.05% | 51.52% |
| Context public, 256px | 39.47% | 24.59% | 100.00% | 24.59% |
| Context public, 512px | 41.57% | 26.24% | 100.00% | 26.24% |
| Context replay_image_holdout, 256px | 64.60% | 47.71% | 99.79% | 47.76% |
| Context replay_image_holdout, 512px | 68.56% | 52.16% | 99.49% | 52.30% |
| Texture + public CNN features: random_forest | 49.59% | 32.97% | 38.04% | 71.24% |
| Texture + public CNN features: hist_gradient_boosting | 48.98% | 32.43% | 35.79% | 77.57% |
| CNN with public replay, 12 epochs, 512px | 69.23% | 52.94% | 95.94% | 54.16% |

## Interpretation and next decision

The conservative replay recipe controls false positives. Larger inference windows improve recall using the same fold checkpoints. Texture-only and hybrid classifiers recover cracks but produce excessive false positives on image 0450. All candidates, including failed ones, are retained above.

All three target photographs depict Erbil Citadel. Images 0450 and 0650 share a photographer; whether structures overlap is unverified. The earlier experiments informed later choices, so these photos are now development data. There is no untouched test set here. Splitting individual regions from these same photos would not fix that limitation.

Annotations cover selected dark crack pixels and selected intact patches, not complete surfaces. Darkness-based annotation assistance can favor texture models. Unknown areas contain unmeasured errors. User review is valuable but does not establish expert-verified damage types; joints, shadows and missing mortar need a consistent annotation rule.

The next data step is additional photographs with fully outlined cracks and all surrounding non-crack pixels reviewed, including joints, shadows and weathered surfaces. Reserve photographs from different structures before further model selection, check exact/near-duplicate and scene overlap, then evaluate once on those complete masks. Additional regions in one existing photograph do not replace this test. No particular number of regions guarantees 80%.

This remains a binary crack pilot. Erosion, vegetation, four-class segmentation and damage severity are not evaluated by these scores.

## Saved models and reproducibility

- `local-results/reviewed-comparison.json`: full comparison and confusion counts.
- Each experiment directory contains its fixed settings, source hashes, saved predictions, metrics and verification records.
- `local-results/reviewed-replay-final-v1/unet-reviewed-replay-final.pt`: separate four-epoch model trained on all three photos. Fold results are not independent scores for this final model.
- Selected twelve-epoch final model: `local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt`. This was trained separately on all three photographs after the fold comparison; it has no independent test score.
- `predict_reviewed_cnn.py`: apply a trusted local CNN checkpoint at native 512px context; output masks are predictions, never approved training labels.
- `predict_reviewed_texture.py`: experimental texture-only inference; this is not the selected neural-network model.
- `train_reviewed_replay.py`, `train_reviewed_replay_long.py`, `train_reviewed_texture.py`, `evaluate_reviewed_context.py`, `evaluate_reviewed_ensemble.py`, `train_reviewed_hybrid.py`: experiment code. Use unused output directories and the recorded environment/input hashes.
- Eleven metric/inference tests passed: eight metric/texture checks and three CNN checks. Independent audits recomputed stored metrics and checked held-out image exclusion, source hashes and ignored labels.

Apply the saved CNN to a local photograph from the repository directory:

```sh
.venv-ml/bin/python predict_reviewed_cnn.py \
  --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt \
  --image path/to/photo.jpg \
  --output local-results/new-photo-prediction
```

The output contains probabilities, a predicted crack mask and provenance. A predicted mask is not a new human-approved label. Exact package versions are recorded in `requirements-reviewed-lock.txt`.

The experiment originally saved its artifacts locally. Selected outputs and the final binary checkpoint are now prepared for distribution; see docs/ASSETS.md. The complete training-reproduction inputs are not all bundled.

## Method references

The separation of model selection from a final test follows [scikit-learn’s evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html). The texture features include local contrast and [OpenCV morphological transforms](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html).
