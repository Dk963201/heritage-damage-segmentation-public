# Automatic label proposals — NOT ground truth

## Decision and quality warning

Existing repository labels cannot be copied onto unrelated photographs. A model must infer proposals on each new image, and a person must check and correct them.

This experiment uses **CLIPSeg** to propose four classes on all 800 candidate photos. **The proposals fail an initial qualitative suitability check and are not approved for supervised research training or test labels.** In a four-image spot check (0001, 0010, 0020, 0030), some photos show landscapes or distant urban views, and natural rock is proposed as eroded masonry. The 800 downloaded files therefore are not yet an established 800-image masonry dataset. The spot check is not a representative accuracy estimate.

The code runs the model locally; images are not sent to a hosted inference API. The pretrained checkpoint is downloaded once. No ground-truth accuracy, IoU or damage severity can be derived from these proposals.

## Repositories considered

- [Dais et al., crack_detection_CNN_masonry](https://github.com/dimitrisdais/crack_detection_CNN_masonry): masonry crack segmentation code and paired example images/masks. Binary crack/non-crack categories do not supply verified erosion, vegetation or healthy-masonry labels on our images. Non-crack does not mean healthy masonry.
- [S2DS](https://github.com/ben-z-original/s2ds): background, crack, spalling, corrosion, efflorescence, vegetation and control-point categories. Background is not healthy masonry; spalling is not an interchangeable erosion annotation. Academic-use restrictions apply. This dataset was not downloaded or copied into this project.
- [Lüddecke and Ecker, CLIPSeg](https://github.com/timojl/clipseg): CVPR 2022 text/image-prompt segmentation. Selected for an exploratory, zero-shot proposal run, not because it is validated for structural damage.

## Model provenance and license notes

Model: [CIDAS/clipseg-rd64-refined](https://huggingface.co/CIDAS/clipseg-rd64-refined).
Pinned revision: `999e0328d9e10b484360c477313983f9afdd7050`.
Loaded with Transformers from `model.safetensors`, without repository-supplied remote code.
The Hugging Face model metadata declares Apache-2.0. The upstream CLIPSeg repository says its MIT code license excludes weights. We record both statements; checkpoint weights are **not redistributed** here. Original image attribution and license information remain in the image release and source manifest. Do not assume model metadata changes any image license.

Citation: Timo Lüddecke and Alexander Ecker, *Image Segmentation Using Text and Image Prompts*, CVPR 2022, pp. 7086–7096.

## Exact proposal settings

| ID | Prompt |
|---|---|
| 0 | healthy intact stone masonry wall |
| 1 | a crack in a masonry wall |
| 2 | eroded crumbling stone surface |
| 3 | vegetation growing on a masonry wall |
| 255 | Unknown / insufficient separation; ignored |

Use the processor's 352×352 input, RGB original image (no CLAHE), sigmoid output for each prompt, highest score ≥0.50 and a gap ≥0.10 over the runner-up. These thresholds are uncalibrated choices, not a validated confidence guarantee. Otherwise assign 255. Output masks are nearest-neighbor expanded to each original image's dimensions. Enlarging them does not restore fine cracks lost at 352px, and scores across prompts need not be calibrated.

For efficiency, cache the text conditions and compute the image encoder once per photo, then repeat its feature maps for four decoder conditions. The first image is compared against the standard four-image-batch forward pass and must agree within 0.001 maximum sigmoid-score difference. This is a numerical compatibility check, not a label-quality check.

## How to run

From the VS Code integrated terminal in the project folder, activate `.venv-ml`, then:

```sh
python -m pip install -r requirements-labels-lock.txt
python auto_label.py
python -m unittest test_metrics test_preprocessing test_pseudo_labels -v
python check_draft_pipeline.py
```

The original ten image ZIPs and download manifest are required; see `LOCAL_RUN.md`. The default processes 800 images. `--limit 20` is available for an exploratory subset, but the complete pipeline validator deliberately requires all 800.

Outputs: `data/processed/clipseg-drafts-v2/0001.png` etc., `pseudo-label-audit.csv`, `pseudo-label-summary.json`, and `pseudo-labels-unreviewed.zip`. The ZIP contains draft masks and metadata, not photographs or pretrained weights. **Never edit masks in this generated folder**: a rerun overwrites them. Copy accepted/corrected versions to a separate reviewed-annotation directory and retain provenance.

## What the repeated tests mean

1. Ten unit tests cover metric arithmetic, preprocessing, patch boundaries and uncertain-label handling.
2. Every generated mask is checked for hash, original-image dimensions and allowed values {0,1,2,3,255}.
3. Each of U-Net, DeepLabV3+ and FPN runs one optimizer step on the same two real photographs with draft masks, ignoring 255. This only verifies the image/mask/loss integration; weights are discarded. It is not full training, a comparison of model performance or an independent test.

## Required next step

Screen all photos for close-up masonry detail, replace unsuitable images, and have a knowledgeable annotator correct the proposals. Keep non-masonry and uncertainty ignored; never convert these to healthy masonry. Manually create and freeze a separate ground-truth evaluation set, group related views before splitting, and then run the actual three-model training comparison. Machine-generated masks must not be silently promoted to test ground truth.

## Independent-label reference check

`python evaluate_external_cracks.py` evaluates the same pretrained CLIPSeg checkpoint with the crack prompt against all 100 author-annotated images in Storey and Snowball's [UK masonry crack dataset](https://zenodo.org/records/18458458), CC BY 4.0. It reuses the verified local ZIPs and does not copy their labels onto different images. No local fitting or threshold tuning uses this reference set. Possible overlap with foundation-model pretraining is unknown, so this is an external reference check rather than proof of uncontaminated generalization.

The ground truth stays at its original dimensions. A 352px prediction thresholded at 0.5 is nearest-neighbor expanded for comparison. The report includes a global binary confusion matrix, crack Dice/IoU and the all-background accuracy baseline. High pixel accuracy alone is misleading when cracks occupy few pixels. This evaluation cannot establish four-class accuracy, Kurdish-site performance, or structural safety.
