# Local Python run and reproducibility

## What this run does

All 800 downloaded candidate images are decoded at their downloaded resolution, checked against the SHA-256 and dimensions in the download manifest, and passed through CLAHE. The run records per-image brightness before/after processing and counts possible 256×256 sliding-window patches with stride 256 and an overlapping final edge patch. Patch size is an implementation choice, not a confirmed teacher requirement. Processed arrays are discarded after measurement to conserve storage; patches and trained weights are not produced.

The image files stay in the ten release ZIPs. The script reads one image at a time, does not extract archives, and does not redownload files. A separate 64-bit difference hash flags pairs with Hamming distance ≤4 for manual visual review. Such flags can be false positives or miss duplicates; they are not proof of duplicate content and no image is automatically removed.

**This is a data/preprocessing result, not a trained damage detector.** There are no pixel annotation masks for these 800 images. Pixel accuracy, mean IoU and class Dice cannot be honestly reported yet. Image brightness and similarity do not measure structural damage.

## Tools used

- Python 3.12 in a project-specific `.venv-ml` environment.
- Pillow: image decoding; NumPy: image arrays; OpenCV: RGB↔LAB conversion and CLAHE (clip limit 2.0, grid 8×8).
- Python standard library: ZIP reading, SHA-256 verification, CSV/JSON outputs.
- PyTorch and segmentation-models-pytorch: U-Net, DeepLabV3+ and FPN software checks, each with an EfficientNet-B0 encoder and four output classes.
- Synthetic model checks use random inputs and random targets, no pretrained weights, Focal + Dice loss, Adam and a five-step linear learning-rate warmup. One optimizer step per model checks code compatibility only. Its loss is deliberately not presented as research performance.
- Exact installed dependency versions: `requirements-ml-lock.txt`. Tested on an Apple M1 Mac with 8GB RAM. Model checks use CPU for portability; MPS availability is recorded separately.

References: [SMP documentation](https://smp.readthedocs.io/en/latest/), [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html), and the assignment reference in `SOURCES.md`.

## Open and run in VS Code

1. Open `Heritage.code-workspace` in VS Code.
2. In the integrated terminal, from the repository folder:

```sh
python3.12 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install -r requirements-ml-lock.txt
python -m unittest test_metrics test_preprocessing -v
python run_local.py
python model_check.py
```

The environment is already installed on the original computer. On another computer, install Python 3.12 first. The lock captures the tested Mac environment; other platforms may require compatible platform-specific wheels.

Download all ten ZIP assets from the [public image release](https://github.com/Dk963201/heritage-damage-segmentation-public/releases/tag/photos-v0.1-candidates) into `data/raw/image-versions/`, retaining their filenames. Keep the root `downloaded-800-manifest.csv`. No extraction is needed. The public release requires no repository invitation.

Alternatively choose **Terminal → Run Task → Audit and preprocess all 800 images**. There is a separate task for the synthetic model check. VS Code workspace tasks target macOS/Linux; on Windows activate the environment and use the terminal commands above with the Windows activation path.

## Outputs

- `local-results/run-summary.json`: actual run counts, duration and explicit training status.
- `local-results/image-audit.csv`: one row per successfully processed image.
- `local-results/similarity-review.csv`: image-ID pairs requiring visual comparison.
- `model-smoke-results.json`: software checks for each architecture, clearly synthetic.
- `test-results.txt`: unit-test record from the initial run.
- `RESULTS.md`: readable record of the initial run. Published CSV/JSON snapshots are also available at repository root; reruns write fresh audit outputs into `local-results/` without overwriting the published root snapshots.

## What must happen before real model results

1. Visually review all candidates for relevant masonry and adequate detail. Replace unsuitable photos and resolve similarity flags. Keep at least 800 suitable annotated images if that minimum remains required; 800 downloaded photos alone do not satisfy it.
2. Annotate each pixel using 0=Healthy Masonry, 1=Structural Crack, 2=Surface Erosion, 3=Vegetation Overgrowth. Mark non-masonry/ambiguous pixels 255 (ignore), subject to instructor approval. Use loss/metric implementations that explicitly ignore 255; the synthetic check has no ignored pixels. Have labels quality-checked.
3. Record structure and capture-session groups. Split groups into training/validation/test **before** extracting patches; do not leak overlapping patches or similar views between splits. Choose proportions based on group/class coverage, then freeze the test split.
4. Implement and verify a mask-aware dataset loader, paired geometric augmentation, image-only color jitter, training/checkpoint selection and full-resolution held-out evaluation. The current scripts do not implement a complete trainer.
5. Train the three models with the same splits and budget. Tune on validation only. Report pixel accuracy, mean IoU and per-class Dice on held-out labelled data, with confusion matrices and failure examples. Do not use this synthetic test as model selection evidence.

No teacher submission, official marking, expert structural safety assessment, or completed supervised model is implied by this run.

## Annotation tracking

Use `annotation-review-template.csv` to track each image's relevance review, structure/capture group, mask filename, label quality check and eventual split. Blank entries mean pending; none have been automatically marked reviewed or labelled. Keep the original source attribution files with the imagery. Non-masonry photographs must not be silently labelled Healthy Masonry.

The initial OpenCV 5 import stalled on this machine. The completed run uses OpenCV 4.12.0.88 and NumPy 2.2.6, as pinned in the lock file. No operating-system security controls were changed.
