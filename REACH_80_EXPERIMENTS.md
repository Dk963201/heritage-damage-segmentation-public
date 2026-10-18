# Attempts to reach 80 percent crack Dice

Date: 18 October 2026. Target means crack foreground Dice on the existing UK masonry validation set. It does not mean four-class Kurdish accuracy.

## Baseline

The best saved run remains `local-results/public-masonry/unet-diagnostic.pt`: 73.2466% pooled crack Dice / 57.7867% IoU at threshold 0.50. It uses 80 UK training sources, 240 public Dais patches in the mixed training pool, and the same 20 fixed UK validation center crops used in previous development runs. The split has been reused and inspected; scores are development evidence, not independent test results. The requested 80% is unconfirmed as an externally validated requirement.

## Threshold calibration

A sweep from threshold 0.10 through 0.90 on the same validation predictions selected 0.35, scoring 73.3288% Dice / 57.8891% IoU. The gain is only 0.08 percentage points. Because the threshold was selected on this validation set, this result is optimistically biased and must not be presented as a new held-out score. The fixed 0.50 checkpoint and score remain the comparable baseline.

## Continued training

Resumed the 73.25% checkpoint for five additional epochs with the existing 400-patch pool, balanced replacement sampler, and the training script's AdamW learning rate 0.00005 with cosine decay toward 0.00001. The optimizer state was freshly initialized; this is a continuation experiment, not a controlled learning-rate ablation.

| Additional epoch | Crack Dice | Crack IoU |
|---:|---:|---:|
| 1 | 66.59% | 49.92% |
| 2 | 68.90% | 52.55% |
| 3 | 69.16% | 52.86% |
| 4 | 65.70% | 48.93% |
| 5 | 67.07% | 50.45% |

All five epochs were below the original checkpoint. A threshold sweep on the final epoch selected 0.90 but reached only 43.83% Dice / 28.06% IoU. That threshold is unstable and should not be used; it exposes strong overfitting or calibration shift on this reused validation set. The original checkpoint remains preserved and is still the best result.

## Additional source examined

FCD-108 describes 108 manually annotated smartphone façade images from Ferizaj, Kosovo, distributed under CC BY 4.0. Its class is binary crack/background, so it can only support crack transfer learning, not erosion/vegetation labels. The linked Google Drive folder opened but contained no visible files in this session; no FCD data were downloaded or used.

## What this means

These attempts did not reach 80%. The next useful step is more diverse, legally reusable masonry and façade crack masks, grouped by building, followed by careful low-rate fine-tuning and an untouched building-level test. The current 20-image UK validation set is too small and reused too often to support a reliable claim that a tuned result generalizes. Proper Kurdish four-class performance still requires at least 800 suitable reviewed target images and a separately agreed severity rubric.
