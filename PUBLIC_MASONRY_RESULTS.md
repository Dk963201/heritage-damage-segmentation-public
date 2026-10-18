# Public-image search and labeled masonry experiment

**Final result: 73.25% crack Dice, up 4.79 percentage points from 68.45%; 80% not reached.**

18 October 2026. User requested public image search to improve segmentation. Ordinary search images lack verified pixel masks, so this experiment uses a research author's public labeled subset. Public search is not Google Photos library access.

## Downloaded and audited source

D. Dais, İ. E. Bal, E. Smyrou and V. Sarhosis, **Automatic crack classification and segmentation on masonry surfaces using convolutional neural networks and transfer learning**, Automation in Construction125(2021),103606. DOI: https://doi.org/10.1016/j.autcon.2021.103606

Source: https://github.com/dimitrisdais/crack_detection_CNN_masonry
Pinned commit: `83c371c70a27852f22ab3386269c6058a203ef38`.

Downloaded240 image/mask pairs (224×224) plus repository LICENSE. Every file verified against its Git blob SHA1; SHA256 and source URL recorded in `public-masonry-download.json`. Repository license is GPL-3.0; no separate dataset license was found. Original files/license remain in `data/raw/dais-public`; they are not relabeled as CC-BY or redistributed in this update. This report credits the source; third-party code was not executed.

All masks contain only0/255 and match the source image dimensions. All240 public patches contain labeled crack pixels; their mean positive pixel fraction is4.21%. Existing UK negative training crops remain in the mixed pool to help constrain false alarms. Decoded RGB duplicate checking found no internal exact duplicates. Screening against all100 original UK photos and their center crops found no exact RGB or64-bit difference-hash match within Hamming distance4. **This does not prove absence of every transformed crop or same-building overlap.** Full scene grouping is unavailable, so UK scores remain development results.

Visual review of12 evenly sampled pairs confirmed alignment and representative brick/plaster cracks. This is a sample review, not expert re-annotation of240 masks; annotation widths and conventions vary. The contact sheet is local at `public-masonry-audit-contact.jpg`.

## Other public sources found

- [YTU-CrackIS](https://zenodo.org/records/19051849): historic church masonry in Cappadocia; source reports1,904 training and469 validation patches, polygon crack labels, and YOLOv8x weights. Relevant heritage-domain follow-up; not downloaded or trained in this run.
- [FCD-108](https://github.com/edonaa/facade-crack-dataset): source reports108 facade photos with manual masks and CC-BY4.0 licensing, download via Google Drive. Potential thin-plaster-crack follow-up; not ingested here.

## Prespecified experiment

Resume the saved68.45%-Dice U-Net checkpoint, train3 additional epochs on160 existing UK training crops plus240 public patches. Resize public RGB224→256 with bilinear interpolation; masks use nearest neighbor. Keep validation's20 UK images and fixed256px center crops unchanged. AdamW5e-5, cosine decay to1e-5, BCE+Dice, training-only paired rotations/flips, frozen BatchNorm running statistics, batch2, CPU2threads, seed42, crack threshold0.5. Balanced replacement sampling means the240 eligible images are a sampling pool; report how many unique public images actually appeared. No new public image is added to UK validation.

Preserve the previous68.45% checkpoint. Report final metrics even if performance decreases; do not call a decrease an improvement or silently select the best epoch. The80% crack-Dice target is distinct from background-dominated pixel accuracy.

```sh
.venv-ml/bin/python fetch_public_masonry.py
.venv-ml/bin/python audit_public_masonry.py
.venv-ml/bin/python train_public_masonry.py --pretrained --balanced --unet-only --epochs 3 --output local-results/public-masonry
.venv-ml/bin/python evaluate_improved_pilot.py --root local-results/public-masonry --baseline local-results/expanded-pilot/unet-diagnostic.pt
.venv-ml/bin/python predict_pilot_sample.py --root local-results/public-masonry
```

The download script uses the pinned `public-dais-tree.json`; existing UK ZIPs, prior checkpoints and environment are required as documented in earlier run guides. New result is not an independent test, four-class result or measurement on the800 Kurdish photos.

## Completed training results

| Pass | Crack Dice | Crack IoU |
|---|---:|---:|
| 1 | 71.08% | 55.14% |
| 2 | 73.03% | 57.52% |
| 3 | 73.25% | 57.79% |

Final saved checkpoint:73.24657766% Dice,57.78666667% IoU. Final confusion matrix [[1269470,4722],[12691,23837]]. This is pooled foreground overlap at fixed0.5 threshold, not four-class accuracy. Pixel accuracy98.67% is background-dominated and is not the target metric.

The pool contained240 eligible public pairs; balanced replacement sampling actually used198 distinct public images during3 epochs. The sampler does not guarantee every file is seen. The prior68.45% checkpoint remains intact. The public-data checkpoint is local at `local-results/public-masonry/unet-diagnostic.pt`; checkpoint weights and public source images are not uploaded in this update.

Because more training and new data were introduced together, this comparison does not isolate the causal contribution of new data from additional optimization. It is a useful improvement experiment, not a controlled ablation.

## Verification and target-domain limitation

A separate inference pass reproduced the final confusion matrix exactly; original validation split/crop equality passed. Full per-image results are in `public-training-comparison.json`.

Kurdish photo0650 was rerun using the final saved model. Visual review shows fewer marked regions than the previous illustration, but proposals remain on some open joints and foreground detail while upper-left plaster crack-like lines are still unmarked. Fewer proposals do not prove better accuracy. No verified target mask exists, so no numerical Kurdish accuracy is reported. Red is unreviewed crack proposal; every other mask pixel is unknown, never certified healthy.

UK validation visualizations are adapted from Storey and Snowball, https://zenodo.org/records/18458458, CC BY4.0. Kurdish0650 attribution is recorded in `annotation-queue-sources.csv`. Public Dais photos/masks and their local contact sheet are not redistributed by this commit.
