# Use the project

First follow [Install](INSTALL.md). Run from the repository root. `python` below means the environment's Python; each command is one line and works in PowerShell or a Unix shell with that replacement. Quote paths containing spaces. Use a fresh output directory for every run.

## A. Inspect what exists

```sh
python project.py status
python project.py --help
```

Status inventories evidence; it does not retrain or measure accuracy. `INCOMPLETE` is expected while four-class labels and the approved severity dataset are missing.

## B. Predict cracks now with the existing binary model

Download and extract the [binary model package](ASSETS.md) into the repository root. It restores `local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt`. Supply your own close, clear wall photo, for example `wall.jpg`:

```sh
python project.py predict-cracks --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt --image wall.jpg --output local-results/my-wall-v1 --device cpu
```

Apple Silicon users can choose `--device mps` if available. The default `auto` selects MPS or CPU; CUDA is not implemented in this predictor.

Open `local-results/my-wall-v1/predicted-crack-mask.png`: white is predicted crack, black is predicted non-crack. Compare at the same dimensions with the original photograph. `probability.npz` stores floating-point probabilities, and `metadata.json` records source/checkpoint hashes, threshold, image dimensions and settings. This command does not produce a color overlay; four-class inference below does.

The default uses 512 × 512 native tiles, ImageNet normalization and threshold 0.5. This is a visual demonstration, not an accuracy test. A photo without a reviewed reference mask cannot yield Dice/IoU. The final checkpoint was trained on all three reviewed photos and has no independent test score. Joints and shadows can be false positives; fine cracks can be missed.

## C. Train and predict the four pixel classes

**This stage needs new reviewed data.** Copy `training-manifest-template.csv` to `training-manifest.csv` and fill it with actual reviewed images/masks and source hashes. Masks use 0 healthy masonry, 1 visible crack, 2 erosion, 3 vegetation, 255 ignore. Sky and unknown areas are not automatically healthy. Keep each building entirely in one split and provide both cities in train, validation and test. See [the annotation/training guide](../FOUR_CLASS_RUN_GUIDE.md).

```sh
python project.py check-segmentation-data training-manifest.csv
python eda_four_class.py training-manifest.csv --output target-eda.json
python project.py train-segmentation training-manifest.csv --output local-results/four-class-target-v1 --epochs 30 --warmup 5
```

This trains U-Net, DeepLabV3+ and FPN sequentially; inspect validation histories and `comparison.json`. Default ImageNet initialization may download encoder weights; `--random-init` avoids that but changes the experiment. Freeze the experimental choices before opting into held-out test evaluation with `--evaluate-test`. Do not repeatedly select models on test scores.

Once a real four-class checkpoint exists:

```sh
python project.py predict-segmentation --checkpoint local-results/four-class-target-v1/unet-best.pt --protocol local-results/four-class-target-v1/protocol.json --model unet --image wall.jpg --roi wall-roi.png --output local-results/my-wall-four-class-v1
```

The checkpoint's `.metadata.json` sidecar and the matching protocol must travel with the weights. `wall-roi.png` must be a same-sized binary mask: 0 excluded, 1 or 255 included. Outputs include the four-class mask, aligned color overlay, probabilities and provenance; see [the exact output contract](../FOUR_CLASS_INFERENCE.md). A wall-area fraction is a pixel measurement, not physical damage area or a severity grade. There is no trained target four-class checkpoint in this delivery.

## D. Train and predict visual severity

**Blocked on an approved rubric and reviewed image-level labels.** Discuss `severity-rubric-draft.json` with the lecturer/conservation reviewer, document an actual approval, then create `severity-rubric-approved.json` and a completed `severity-manifest.csv`. The draft's grades are proposed, not official.

```sh
python project.py check-severity-data severity-manifest.csv --rubric severity-rubric-approved.json
python project.py train-severity severity-manifest.csv --rubric severity-rubric-approved.json --output local-results/severity-target-v1 --epochs 30
python project.py predict-severity --checkpoint local-results/severity-target-v1/severity-best.pt --image wall.jpg --output local-results/my-wall-severity-v1
```

Check the saved filename in [Severity workflow](../SEVERITY_WORKFLOW.md) and use that run's checkpoint. The classifier starts randomly initialized, learns from independently reviewed labels, and outputs an uncalibrated probability distribution. It cannot determine whether a building is structurally safe. No target severity model or severity-accuracy result is bundled.

## E. Review and improve

The pipeline package includes `local-results/two-site-review-20261018/contact-sheet.jpg`, `review.html`, credits and nine **pending** source candidates. Open them locally for review; thumbnails alone are insufficient for final pixel masks. Follow [Improving results](IMPROVING_RESULTS.md) before another training round. Do not turn model predictions into accepted labels without review.
