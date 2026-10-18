# Automatic labelling experiment: quality check FAILED

Generated draft masks for all **800 candidate images** using a pretrained CLIPSeg model, then reran data and model-integration checks. **Do not use these masks as ground truth.** The model failed an external annotated crack benchmark and initial visual checks found unsuitable candidate photographs and false erosion labels.

## Results

| Check | Result |
|---|---:|
| Draft masks generated | 800 |
| Mask hash, dimensions and label-ID checks | 800 passed |
| Pixels receiving a proposed class | 12.97% |
| Pixels left unknown (255) | 87.03% |
| Human-verified complete masks | 0 |
| Unit tests | 10 passed |
| U-Net / DeepLabV3+ / FPN integration | All three passed one optimizer step on two real images with draft masks |
| Full supervised training | Not performed |
| Kurdish four-class accuracy | Unknown |

## Real reference test on 100 externally annotated images

Source: Storey and Snowball, [UK masonry crack images and human annotations](https://zenodo.org/records/18458458), CC BY 4.0. This is a **binary crack test on a different dataset**, not a test of the four Kurdish classes. No local fitting or threshold tuning used these 100 images. Foundation-model pretraining overlap is unknown.

| Metric | Result |
|---|---:|
| Crack Dice | 1.20% |
| Crack IoU | 0.60% |
| Pixel accuracy | 71.67% |
| Accuracy from predicting every pixel as background | 99.15% |

The low crack overlap and large false-positive count make this configuration unsuitable for automatic ground-truth labelling. Background dominates the images, so even the much higher all-background accuracy is not a useful crack detector. No threshold was tuned to make the results look better.

Initial visual checks of image IDs 0001, 0010, 0020 and 0030 also found natural rock mistaken for erosion and landscape/distant town views with insufficient masonry detail. These four checks are not a representative dataset-accuracy estimate. See [review notes](draft-visual-review.csv).

## Files and reproduction

- [800 UNREVIEWED mask proposals (ZIP)](pseudo-labels-unreviewed.zip), [archive checksum](PSEUDO_LABEL_SHA256.txt)
- [Per-image mask audit](pseudo-label-audit.csv), [proposal settings and counts](pseudo-label-summary.json)
- [Reference benchmark results](external-crack-results.json), [per-image reference confusion counts](external-crack-per-image.csv)
- [Three-model integration checks](draft-pipeline-results.json), [unit-test output](pseudo-label-tests.txt)
- [Sources, method, exact prompts, limitations and run commands](AUTO_LABELS.md)

The ZIP includes original source/attribution metadata. It contains no pretrained weights. Retain the original image licensing terms.

**Next required work:** screen and replace unsuitable photographs; create or obtain qualified human-reviewed masks and an independent evaluation set. The machine proposals are retained for inspection only. Training a student model on these unchecked masks would teach it the labeller's errors; scoring against those same masks would measure agreement, not true damage accuracy.
