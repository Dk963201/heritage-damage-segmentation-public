# Resources and 80% target experiment — 18 October 2026

Target: 0.80 pooled foreground crack Dice at threshold 0.5 on the existing 20 UK validation center crops. This is a development target, not Kurdish four-class accuracy or an untouched test result. Keep the exact 80/20 source split and crop coordinates; do not remove difficult validation images or train on validation labels.

## Additional primary resources inspected

- [OmniCrack30k](https://github.com/ben-z-original/omnicrack30k): CVPR Workshops 2024 project, pretrained nnU-Net inference and evaluation. Consolidated training data requires an educational request and adherence to component licenses; no request was sent. Useful future external pretrained baseline; not installed or claimed as tested here.
- [Annotated crack dataset collection](https://github.com/preethamam/Cracks-Semantic-Segmentation-Datasets): lists an 81-image high-resolution masonry/brick set with human segmentation labels (5152×3864), including concrete joints. Potential training expansion; inspect original component licenses and deduplicate against our validation before use. Not ingested in this run.
- [MCrack1300 paper](https://arxiv.org/abs/2401.15266): masonry instance segmentation data covering bricks, broken bricks and cracks. Different taxonomy from our healthy/crack/erosion/vegetation task. Useful for learning to distinguish joints and damage; not ingested in this run.
- [Road U-Net reference](https://github.com/yakhyo/crack-segmentation): compares cross entropy, Dice, combined and focal losses. Its README warns its reported Dice calculation may not fairly compare models. Do not import its headline scores as our accuracy.
- [OpenCrack](https://github.com/fadeevla/OpenCrack): composite benchmark and pretrained-model reference, emphasizing duplicate control and cross-domain degradation. Source roster includes masonry; checkpoint training overlap would need auditing before a fair comparison. Not tested here.
- [SMP loss documentation](https://segmentation-models-pytorch.readthedocs.io/en/latest/losses.html): binary Dice and focal definitions.

## Experiment fixed before results

Resume the prior saved epoch-5 U-Net/ EfficientNet-B0 checkpoint. Train 20 additional epochs. AdamW 1e-4, weight decay 1e-4, cosine decay to 1e-5; binary cross entropy + Dice; gradient clipping norm 1; training-only paired 90-degree rotations and flips; freeze BatchNorm running statistics to reduce small-batch instability. Keep balanced sampling, ImageNet input normalization, batch size 2, CPU two threads, seed42. Validate at fixed threshold0.5 and report ALL epochs plus the final checkpoint. Several settings change together, so this is an improvement attempt, not an ablation proving which setting caused a change.

An initial attempt stopped before completing epoch1 because rotated tensors were noncontiguous. Fixed with contiguous tensors before computing the loss and restarted from the same original checkpoint and seed.

No new external photos or labels entered the training/validation split. Do not claim that resource discovery means those models were run. The 800 Kurdish photos still need verified target masks.

## Reproduction

```sh
.venv-ml/bin/python train_improved_pilot.py --pretrained --balanced --unet-only --epochs 20 --output local-results/improved-pilot
```

Requires the original audited UK ZIPs, pilot-audit.json, existing environment, ImageNet weights, and prior local checkpoint `local-results/binary-balanced/unet-diagnostic.pt`. Regenerate that checkpoint with the prior PILOT_RUN_GUIDE.md if absent. Checkpoints stay local; results and scripts can be published.
