# Supervised pilot results

Completed 18 October 2026. **Real training completed on this Mac. The model is not ready for automatic annotation or heritage inspection.**

## Scope and data

100 published UK masonry images with author-provided binary crack masks (https://zenodo.org/records/18458458, CC BY 4.0). Fixed 256px center crops: 80 train, 20 validation. Only 22 training crops and 9 validation crops contain crack pixels. No Kurdish four-class ground truth was used. Masks are source-provided human annotations, not independently re-annotated here.

## Initial workflow experiment

U-Net, FPN and DeepLabV3+ with randomly initialized EfficientNet-B0 encoders each completed 3 epochs. All three ended at **0% crack Dice**, predicting all background on validation. Background-only pixel accuracy is 97.21%; this is not evidence of useful detection. These short runs demonstrate functioning training, not a fair ranking of trained architectures.

## Revised U-Net

ImageNet-pretrained EfficientNet-B0, proper ImageNet normalization, weighted sampling of training crops to balance crack presence, 5 epochs. Same validation images, threshold 0.5. Several settings changed together; no causal ablation claim.

| Epoch | Crack Dice | Crack IoU | Pixel accuracy |
|---|---:|---:|---:|
| 1 | 61.39% | 44.29% | 97.84% |
| 2 | 38.53% | 23.87% | 97.83% |
| 3 | 32.23% | 19.21% | 93.11% |
| 4 | 17.75% | 9.74% | 96.33% |
| 5 | 52.82% | 35.89% | 97.57% |

The saved checkpoint is epoch 5, not the best-scoring epoch. The fluctuating curve shows instability. These are development scores: building groups are unavailable, scene leakage cannot be ruled out, and the public dataset was previously inspected for the earlier CLIPSeg benchmark. No independent test accuracy or Kurdish transfer accuracy is claimed.

## Deliverables

- Training script and VS Code task: `train_binary_pilot.py`, `Heritage.code-workspace`.
- Reproduction/settings: `PILOT_RUN_GUIDE.md`, encoder revision/hash in `pretrained-encoder-provenance.json`.
- Full histories, confusion matrices and split: `binary-pilot-results.json`, `binary-balanced-results.json`, `binary-pilot-split.json`, `binary-crop-audit.json`.
- Local saved checkpoints: `local-results/binary-pilot/` and `local-results/binary-balanced/unet-diagnostic.pt`. Checkpoints are local, not uploaded in this commit.
- 37-photo Kurdish review pack: `annotation-review-pack.zip`; 29 candidates, 1 exposure hold, 7 context-only. Not 37 accepted labels.
- Four-class manifest validator rejects unapproved masks, incorrect source hashes, invalid labels, missing classes and cross-split building groups. Five tests pass. It cannot certify mask correctness.

## Required next work

Review and correct target-domain pixel masks, resolve ambiguous class definitions with the supervisor, identify building groups, and create an untouched test set. Then fine-tune the requested three architectures using the same target splits and full planned recipe. The required 800 annotated images remain incomplete. Do not turn this binary model’s non-crack pixels into healthy masonry, and do not use its predictions as verified labels.

## Kurdish transfer illustration (photo 0650)

The saved epoch-5 model was applied to a 1024px preview using overlapping 256px tiles. Visual inspection of `0650-draft-overlay.jpg` shows proposed crack pixels along several horizontal masonry joints/missing-mortar lines, while thin crack-like lines in upper-left plaster are missed. This is qualitative evidence of transfer errors, not measured accuracy: no verified pixel mask exists for this photo. Red proposals must not be accepted automatically. All other mask pixels remain unknown (255), not healthy masonry. Image attribution is recorded in `annotation-queue-sources.csv`; inference settings/checkpoint hash are in `0650-transfer-check.json`.
