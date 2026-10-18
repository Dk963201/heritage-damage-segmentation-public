# Photo screening pilot — blocked before inference

Moondream2 did **not** produce a prediction. Do not interpret this pilot as a successful installation of an operational plugin, a quality score, or completed screening of the 800 photos.

We prepared ten photos and recorded assistant visual judgments before attempting inference. These are provisional scene judgments, not expert ground truth or pixel damage masks. The purposive sample is not representative of all 800 images.

## What was done

- Installed FiftyOne 1.22.1 and Transformers 4.49.0 in an isolated `.venv-screen`; verified FiftyOne imports successfully after pinning setuptools below 81.
- Inspected the community [Moondream2 zoo integration](https://github.com/harpreetsahota204/moondream2) at commit `86b69478a5494683d7aa7cc70eac15380750f07f`.
- Downloaded the [Moondream2 model](https://huggingface.co/vikhyatk/moondream2) at revision `8ef2caddfdead307fe7f3a4ed596baca7eab8a2f` (2025-03-27).
- Prepared `screen_sample.py`, a direct model adapter using the same model/API, explicit float16/MPS loading and the plugin's missing-module workaround. The community wrapper itself was not registered or run end to end. No photo was sent to a hosted model API.
- Attempted local inference on an Apple M1 with 8 GiB unified memory. Loading caused severe resource pressure; free disk fell to 254 MiB before the first prediction. Stopped the process and removed only the reproducible 3.85 GB downloaded weight file. Free disk recovered to 5.3 GiB. Dataset and previous experiment outputs were preserved.

## Provisional visual review

| Photo | Assistant reference | Reason |
|---|---|---|
| 0001 | Uncertain | Carved rock relief, not clearly coursed masonry |
| 0010 | Building overview | Small distant buildings on a hill |
| 0030 | Landscape/other | Mountains and valley |
| 0192 | Building overview | Wide courtyard view |
| 0250 | Building overview | Street and buildings |
| 0350 | Building overview | Building above rubble/earth slope |
| 0450 | Building overview | Courtyard facade and arches |
| 0550 | Building overview | Citadel building on a slope |
| 0650 | Close-up masonry candidate | Brick wall and vegetation |
| 0750 | Building overview | Facade and windows |

Overview photos may contain usable crops. No image has been deleted or definitively rejected by this review. Damage classes remain unverified.

## Reproduce on suitable hardware

Use Python 3.12 and `requirements-screen-lock.txt` in a new environment. The tested local environment shares existing `.venv-ml` packages via a `.pth` file to avoid duplicate Torch installations; a fresh machine should install the full lock normally. The scripts currently target Apple MPS. Other accelerators require an explicit device adjustment and separate validation.

```sh
python3.12 -m venv .venv-screen
.venv-screen/bin/pip install -r requirements-screen-lock.txt
.venv-screen/bin/python prepare_screening.py
.venv-screen/bin/python screen_sample.py
```

Download the project's existing candidate-photo release ZIPs into `data/raw/image-versions` first. Preparation extracts the ten selected photos and downloads pinned model files. Inspect custom model code before permitting `trust_remote_code`. New safeguards require at least 12 GiB RAM and 12 GiB free disk before downloading, and 8 GiB free disk before inference. These are conservative project safeguards, not verified minimum vendor specifications or a guarantee of success.

The run saves each completed result immediately to `screening-results.json`. A successful future run must compare predictions with the pre-recorded references, inspect disagreements, and validate a larger independently reviewed sample before batch use. Screening cannot replace human-checked four-class pixel annotations. The earlier CLIPSeg quality failure remains in effect.
