# Four-class prediction

`predict_four_class.py` runs a saved four-class U-Net, DeepLabV3+ or FPN checkpoint produced by `train_four_class.py`. It is a software inference path. No reviewed four-class target checkpoint is supplied by this work, and the existing binary checkpoint cannot be substituted. Missing or incompatible weights produce an error.

Once a real four-class run has produced matching weights and a protocol:

```sh
.venv-ml/bin/python predict_four_class.py \
  --checkpoint local-results/four-class-v1/unet-best.pt \
  --protocol local-results/four-class-v1/protocol.json \
  --model unet \
  --image path/to/photo.jpg \
  --roi path/to/masonry-roi.png \
  --output local-results/four-class-prediction-001
```

The paths above describe future real training outputs; they are not evidence that those outputs already exist. Use `--model deeplabv3plus` or `--model fpn` with the corresponding checkpoint from the same training run. The checkpoint must be a local tensor `state_dict`; loading uses `torch.load(..., weights_only=True)` and strict architecture matching. EfficientNet-B0 is constructed with `encoder_weights=None`, so inference does not download pretrained weights.

New training runs also write an adjacent sidecar such as `unet-best.metadata.json`. When present, inference requires its checkpoint hash, protocol hash, architecture and class count to match. Pairing same-architecture weights with another run's CLAHE protocol is rejected. Keep this sidecar with the checkpoint. Legacy checkpoints without a sidecar remain usable, but output metadata explicitly says **association unverified**: architecture compatibility alone cannot establish matching preprocessing. Hash agreement proves file association, not human-label quality or scientific accuracy.

The source must be an 8-bit RGB image, at most 25 million pixels. Pixel dimensions and stored orientation remain unchanged; EXIF rotation is not automatically applied, matching training. Convert other image modes explicitly before use. The optional ROI must have identical dimensions and be a nonempty, single-channel binary image encoded 0/1 or 0/255. Include only the relevant masonry surface. This tool does not verify whether the ROI is suitable or modify it.

Without `--roi`, every image pixel receives one of the four classes, including sky, objects and other background. Resulting counts are **whole-image counts**, not wall damage measurements. With an ROI, the class fractions use only included pixels, the class mask is 255 outside it, and probabilities outside it are NaN. No heuristic automatically selects a wall or assigns severity.

## Outputs

- `prediction-classes.png`: original-size single-channel class IDs 0 healthy masonry, 1 visible crack, 2 surface erosion, 3 vegetation overgrowth; 255 outside a supplied ROI.
- `prediction-probabilities.npy`: float32 array `[4, height, width]` in the same class order. Values are **uncalibrated softmax probabilities**, not validated confidence or physical risk. They sum to 1 within the included region.
- `prediction-overlay.png`: source-aligned RGB overlay, 45% class color blended with the original. Outside a supplied ROI, original pixels remain unchanged. Colors are green for healthy masonry, red for visible crack, amber for erosion and purple for vegetation; exact RGB values are recorded in metadata.
- `metadata.json`: source/checkpoint/protocol/ROI hashes, input dimensions, architecture, preprocessing, class counts/fractions, color legend, software versions and limitations.

Outputs are explicitly model predictions, not reviewed ground truth. Descriptive pixel fractions do not establish physical wall area, crack width, structural safety or damage severity. The legacy training name `Structural Crack` is retained in provenance; the displayed class is **Visible crack**, avoiding an unsupported structural diagnosis. No severity grade is assigned.

## Consistency with training

Inference reuses the trainer's `tensor()`, `starts()` and architecture map. Patch size and CLAHE settings come from the supplied protocol; missing or inconsistent settings are rejected. Tiles remain at native resolution, small images receive the trainer's PIL black padding, overlapping **logits** are averaged, and softmax/argmax are applied afterwards. ImageNet mean/std normalization matches training. There is no threshold search, resizing, postprocessing, mask cleanup or new-label creation.

Use a fresh output directory each time; existing results are never overwritten. The CPU implementation stores full-image logits and probabilities, so large images require substantial RAM. The 25-million-pixel limit bounds input size but is not a promise of low memory usage.

## Software verification

```sh
.venv-ml/bin/python -m unittest discover -s tests -p 'test_predict_four_class.py' -v
```

Tests compare predictions with the existing training evaluator on overlapping edge tiles and small padded inputs, with and without CLAHE. They check probability normalization, ROI-only denominators and exclusions, overlay alignment, invalid protocols/ROIs, missing or mismatched weights, swapped preprocessing protocols, checkpoint-sidecar checks and overwrite protection. A temporary random U-Net checkpoint exercises the CLI without downloading weights. All smoke artifacts are deleted afterwards. These tests demonstrate software behavior only and produce no research performance claim.
