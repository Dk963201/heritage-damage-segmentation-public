# Four class training guide

Status: implemented and software-tested. No reviewed Kurdish four-class manifest exists yet; no target-domain training or scores are claimed.

## Prepare real annotations

1. Review the 800 candidates for relevant visible masonry, duplicates, site identity and licensing. Replace irrelevant photographs; 800 downloaded candidates do not meet the requirement for 800 annotated usable photographs.
2. Use original resolution and record masks with IDs 0 healthy masonry, 1 structural crack, 2 surface erosion, 3 vegetation overgrowth, and 255 unknown/outside the task. See ANNOTATION_PILOT.md. Do not label uncertain non-crack pixels healthy or accept generated proposals without review.
3. Record annotator and reviewer, source SHA256, site (Erbil/Amedi) and building group in a copy of training-manifest-template.csv. Assign entire buildings to train, validation or test before patches. Review visual near-duplicates as well as exact hashes. A building cannot belong to both cities. The validator checks fields, not whether a human actually reviewed a mask.
4. Require at least 800 usable reviewed source images for assignment completion. A smaller reviewed pilot is allowed by the software for development but does not satisfy the assignment minimum.

## Run

Use the existing .venv-ml environment; requirements-ml-lock.txt records its packages. It includes torch, segmentation-models-pytorch, OpenCV, Pillow and numpy. Work from the repository directory.

```sh
.venv-ml/bin/python project.py check-segmentation-data training-manifest.csv
.venv-ml/bin/python eda_four_class.py training-manifest.csv --output target-eda.json
.venv-ml/bin/python project.py train-segmentation training-manifest.csv --output local-results/four-class-v1 --epochs 30 --warmup 5
```

Defaults are proposed engineering settings, not confirmed teacher-prescribed values: 256-pixel patches, EfficientNet-B0 encoders, ImageNet initialization, batch 2, CPU, seed 42, learning rate 0.0001, five-epoch linear warmup then constant rate, 30 epochs. All three required architectures run in sequence. ImageNet initialization may download weights. Random initialization is available with --random-init. Training uses paired flips and image-only brightness/color jitter; CLAHE is on by default and can be disabled for a controlled comparison with --no-clahe.

Each run saves the protocol, source/mask hashes, validation history, best-validation checkpoint and architecture comparison. A checkpoint metadata sidecar binds each saved model to its exact protocol hash. Existing output directories are rejected. Validation reconstructs full source images by averaging overlapping tile logits and counts each annotated pixel once. Unknown pixels are excluded. Best checkpoint selection uses validation mIoU only. Results include class support, per-image metrics and separate site summaries. The main project entry point requires both cities in every split; the lower-level trainer's optional --require-sites flag exists for compatibility with explicitly synthetic software fixtures. Test evaluation is off by default; add --evaluate-test only after freezing the experimental protocol. Do not repeatedly tune to test results. Absent classes are reported as null, not perfect scores. See FOUR_CLASS_INFERENCE.md for prediction using the trained checkpoints.

The batch loader drops an incomplete final training batch to avoid single-example batch normalization. Full-source validation accumulates four float32 maps plus a count map in RAM; large photos and 141122 candidate patches make a complete CPU run expensive. Inspect data size before choosing epochs; do not confuse a tiny software test with training on all 800.

```sh
.venv-ml/bin/python -m unittest test_four_class test_metrics test_training_manifest
.venv-ml/bin/python smoke_four_class.py
```

The second command creates temporary synthetic fixtures, runs all architectures through training and checkpoint evaluation, and writes four-class-software-check.json. Its temporary data and checkpoints are deleted. It produces no scientific accuracy claim.

## Remaining research work

Complete reviewed annotations and pixel balance EDA, freeze a grouped experimental protocol, train the models, inspect errors by site/class, report held-out metrics and repeat across seeds or grouped folds where feasible. The current implementation supplies one fixed split, not automated cross-validation. Severity classification requires an expert-defined rubric and labeled severity outcomes; class-area fractions alone cannot establish structural safety.
