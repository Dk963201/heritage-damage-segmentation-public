# Assets, sources and downloads

The project has four different asset types: **photographs** to analyze, **reviewed labels** to learn/check against, **code** implementing the method, and **model weights** learned during training. Having 800 photos does not mean having 800 labeled examples or a trained four-class system.

## What to download

| Asset | Download/location | Use and limitation |
|---|---|---|
| Current source and handbook | Repository **Code → Download ZIP**, or [pipeline package](../heritage-four-part-pipeline-20261018.zip) | Python implementation, templates, tests and instructions. The pipeline package also preserves nested evidence/review folders. |
| Trained binary crack model | [heritage-binary-model-20261018.zip](../assets/heritage-binary-model-20261018.zip) | Existing U-Net/EfficientNet-B0 final checkpoint, protocol and results. Inference on a supplied wall photo; no four-class or severity weights. |
| Reviewed labels and selected evidence | [reviewed-labels-evidence-20261018.zip](../assets/reviewed-labels-evidence-20261018.zip) | Three sparse binary mask snapshots, review provenance and selected result records. These are development labels, not full four-class truth. |
| All 800 candidate photographs | [photos-v0.1-candidates release](https://github.com/Dk963201/heritage-damage-segmentation-public/releases/tag/photos-v0.1-candidates) | Ten ZIPs: `heritage-versions-01.zip` through `10.zip`, 80 photos each, total 2,023,945,223 bytes (about 2.02 GB). Distributed as separately credited photo assets; the exact archive hashes are recorded below. |
| Candidate metadata | [downloaded-800-manifest.csv](../downloaded-800-manifest.csv), [photo checksums](../SHA256SUMS.txt) | Image dimensions, hashes, source links, authors and licenses; collection counts 609 Erbil / 191 Amedi still need relevance/location review. |
| Pending two-city review | Inside the pipeline ZIP: `local-results/two-site-review-20261018/` | Nine source thumbnails, contact sheet, review HTML and source records. All pending; not new approved masks. |
| Exact local environment records | [ML lock](../requirements-ml-lock.txt), [reviewed-experiment lock](../requirements-reviewed-lock.txt) | Historical package versions. Start a new platform with [Install](INSTALL.md), then record its resolved environment. |

To download a ZIP stored as a repository file, open its link and use GitHub's **Download raw file** control. This public copy can be read without signing in. GitHub's automatically generated **Source code (zip)** on the photo release contains source at that tag, **not** the 800 photo assets.

## Extract to the correct place

1. Extract the source/pipeline package into a new project directory.
2. Extract the binary model ZIP **inside that directory**, retaining its `local-results/...` folders. Do not create an extra nested project folder.
3. If needed, extract the reviewed-label/evidence ZIP there too. Inspect its internal README and avoid replacing existing annotations in a working project.
4. Download the ten photo ZIPs and keep the archives in `data/raw/image-versions/` if you want to rebuild the two-site review pack. To browse photos, extract all ten into one separate destination, for example `data/raw/candidates/`. Preserve `images/` and `attribution/` together. Each archive's IDs are unique.

Windows: **Extract All** for source/assets; combine the numbered archive contents in one destination. macOS: use Archive Utility and combine `images`/`attribution` contents without losing existing files. Linux: use the desktop archive manager or `unzip` into a chosen destination. Preserve exact file names and confirm checksums before use.

The three reviewed images can be restored from release archives: **0450** is `images/0450.jpg` in ZIP06, **0650** in ZIP09, and **0750** in ZIP10. Corresponding `attribution/{id}.json` files identify sources and licenses. The research scripts expect copies at `data/processed/review-{id}/{id}-source.jpg`; moving a photo does not change its labels or authorize a different split.

## Verify files

The [asset catalog](../assets/asset-catalog.json) records bytes and SHA-256 for the distributed model and label packages. [Asset checksums](../assets/SHA256SUMS.txt) are distinct from the [photo-release checksums](../SHA256SUMS.txt). The selected checkpoint itself must have SHA-256:

```text
757e6d8eac1f1d661dbbc7e698bdfc2890eca9388b7c6e66cd89e71797b02d21
```

Run this from the project root on any OS with the installed environment Python:

```sh
python -c "import hashlib,pathlib; p=pathlib.Path('local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt'); print(hashlib.file_digest(p.open('rb'),'sha256').hexdigest())"
```

For a photo archive, replace the path with its ZIP path and compare with the release checksum. A matching hash confirms file identity, not accurate labels or model performance. Load only checkpoints from a source you trust; these scripts use weights-only PyTorch loading.

## What we used and why

| Resource | Contribution |
|---|---|
| Python, PyTorch and segmentation-models-pytorch | Training, loading checkpoints, CNN implementations and inference |
| U-Net + EfficientNet-B0 | Existing binary crack prototype; encoder captures visual features and decoder returns pixel predictions |
| U-Net, DeepLabV3+ and FPN, each with EfficientNet-B0 | Implemented controlled four-class architecture comparison; real target training remains pending |
| OpenCV, NumPy and Pillow | CLAHE in four-class processing, image/mask I/O, arrays, tiling and image checks |
| Wikimedia Commons candidate sources | Credited Erbil/Amedi image versions; not expert-labeled damage data |
| Public UK masonry masks and public Dais masonry data | Earlier supervised binary initialization/replay experiments; different scenes and labels from the target cities |
| User-reviewed regions 1–333 | 177,493 reviewed pixels across three Erbil photos: 165,275 intact and 12,218 crack; remaining pixels ignored |
| Source SHA-256, manifests and protocols | Traceability of inputs, labels, settings, splits and model outputs |
| unittest and synthetic fixtures | Verify software behavior without pretending synthetic scores measure heritage accuracy |

Source attribution and public dataset details are recorded in [PHOTO_SOURCES.md](../PHOTO_SOURCES.md), [PUBLIC_MASONRY_RESULTS.md](../PUBLIC_MASONRY_RESULTS.md) and [PILOT_RUN_GUIDE.md](../PILOT_RUN_GUIDE.md). Preserve each photograph's author/license and ShareAlike conditions where applicable. Erbil and Amedi's different UNESCO statuses do not make these files a UNESCO-provided training dataset.

## Reproduction boundaries

**Prediction is ready:** source + installed dependencies + the distributed binary weight + a photo. See [Usage](USAGE.md).

**Exact historical retraining needs more:** the original public initialization checkpoint, fixed UK archive pair and split, cached public replay pool/crop manifest, target source photos, frozen masks and pending-region exclusion records. They are not all bundled in the small code/model download. `train_reviewed_replay_long.py` checks those dependencies and their hashes; it is an experiment record, not a one-command fresh-download training recipe. See [the recorded training report](../REVIEWED_IMPROVEMENT_RESULTS.md).

**Four-class/severity research is not ready:** neither approved target four-class masks nor approved severity data exists yet. Empty templates and pending review files must not be presented as training data. The code supplies a path to train these tasks once the [required review work](IMPROVING_RESULTS.md) is done.
