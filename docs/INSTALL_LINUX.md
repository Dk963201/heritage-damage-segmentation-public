# Linux setup — Heritage Damage Study

This guide takes you from a public download to a crack prediction on your own wall photo. The concrete package commands target **Ubuntu 24.04 LTS, 64-bit Intel/AMD (`x86_64`)**, using Python 3.12 and CPU inference. No GPU is needed. Other distributions and ARM machines need their own Python packages and compatible PyTorch wheels.

**Linux has not been executed for this delivery.** The [fresh-install check](FRESH_INSTALL_CHECK.md) was run on macOS. These Linux instructions were checked against the repository commands and official platform documentation; they are not a Linux test result.

Run one step at a time in Terminal. Stop and resolve an error before continuing. Commands below use the environment's Python explicitly, so activation is optional.

## 1. Install Ubuntu prerequisites

Check the operating system and architecture:

```bash
cat /etc/os-release
uname -m
```

On Ubuntu 24.04, install Python, its virtual-environment support and download/extraction tools:

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv curl ca-certificates unzip git
python3.12 --version
```

The final command should show `Python 3.12.x`. Ubuntu supplies [`python3.12-venv`](https://packages.ubuntu.com/noble-updates/python3.12-venv) through its Universe repository. Use Ubuntu's [package-management guidance](https://github.com/canonical/ubuntu-server-documentation/blob/main/docs/how-to/software/package-management.md) if your configured repositories cannot find it. Do not replace the operating system's Python or use `sudo pip`.

On another Linux distribution, install Python 3.12 and its `venv` support using that distribution's documented method, then continue at step 2. These `apt` package names are specific to Ubuntu; they are not commands for every Linux distribution.

## 2. Download a fresh project

Choose **one** method. Use a new directory so existing labels and prediction outputs are preserved.

### Option A: source ZIP

```bash
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
mkdir heritage-damage-linux
cd heritage-damage-linux
curl --fail --location --retry 3 https://github.com/Dk963201/heritage-damage-segmentation-public/archive/refs/heads/main.zip --output source.zip
unzip source.zip
cd heritage-damage-segmentation-public-main
```

### Option B: Git

```bash
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"
git clone https://github.com/Dk963201/heritage-damage-segmentation-public.git
cd heritage-damage-segmentation-public
```

The project directory must contain `project.py` and `requirements-core.txt`. Keep the terminal in this directory for the remaining steps.

## 3. Create an isolated environment and install dependencies

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python -m pip check
```

The last command should report `No broken requirements found`. The explicit CPU wheel index follows the [PyTorch installation selector](https://pytorch.org/get-started/locally/) for Linux/Pip/Python/CPU. The project uses `opencv-python-headless`; a graphical OpenCV window is not required. The [Python virtual-environment documentation](https://docs.python.org/3.12/library/venv.html) explains why `.venv/bin/python` works without activating the environment.

The historical `requirements-*-lock.txt` files record earlier environments. Start with `requirements-core.txt` here, then record your own resolved versions in step 4. Installing a CUDA wheel would not add GPU support to the project's CPU training commands or binary predictor.

## 4. Check that the installation works

```bash
.venv/bin/python -c "import sys, torch, segmentation_models_pytorch as smp, cv2, numpy; print('Python', sys.version.split()[0], 'torch', torch.__version__, 'smp', smp.__version__, 'OpenCV', cv2.__version__, 'NumPy', numpy.__version__, 'CUDA', torch.cuda.is_available())"
.venv/bin/python project.py status
.venv/bin/python project.py --help
.venv/bin/python -m unittest test_four_class test_metrics test_training_manifest
.venv/bin/python -m unittest discover -s tests -p test_predict_four_class.py
.venv/bin/python -m unittest discover -s tests -p test_project_entrypoint.py
.venv/bin/python -m unittest discover -s tests -p test_severity_pipeline.py
.venv/bin/python -m unittest discover -s tests -p test_two_site_review.py
```

These five test commands select **44 self-contained tests** in the current source. Every group should finish with `OK`. They do not need the 800 photos or a trained checkpoint. `CUDA False` is expected for this CPU installation. `project.py status` reporting incomplete research is also expected: software checks do not create reviewed four-class labels or an approved severity dataset.

Optionally run synthetic training and inference through U-Net, DeepLabV3+ and FPN:

```bash
.venv/bin/python smoke_four_class.py
```

This writes `four-class-software-check.json`, replacing that evidence file in your local copy. Its synthetic scores are not heritage-image accuracy. Record the installed package versions after the checks:

```bash
.venv/bin/python -m pip freeze > installed-environment.txt
```

## 5. Download and verify the trained binary model

The source ZIP contains code and an `assets` folder; the model can also be downloaded separately with the commands below. You do **not** need all 800 photos to predict your own image.

```bash
mkdir -p downloads
curl --fail --location --retry 3 https://raw.githubusercontent.com/Dk963201/heritage-damage-segmentation-public/main/assets/heritage-binary-model-20261018.zip --output downloads/heritage-binary-model-20261018.zip
printf '%s\n' 'c5f1688bce81235bfb6b8dcec74fdeb8c40912e9723daf2848380f5d9181b3a8  downloads/heritage-binary-model-20261018.zip' | sha256sum --check
```

Continue only if the checksum command prints `OK`. Then extract into the project root, retaining the archive's directory structure:

```bash
unzip -n downloads/heritage-binary-model-20261018.zip -d .
printf '%s\n' '757e6d8eac1f1d661dbbc7e698bdfc2890eca9388b7c6e66cd89e71797b02d21  local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt' | sha256sum --check
```

Again, require `OK`. `unzip -n` preserves existing files. The [asset guide](ASSETS.md) lists these hashes, the other downloads and their licensing/provenance. If an existing checkpoint fails verification, extract the verified package into a fresh project directory rather than overwriting experimental work.

## 6. Predict cracks in your wall photo

Copy a clear, close wall photograph into the project directory and name it `wall.jpg`. Keep an original copy. If your photo is a PNG, keep its `.png` extension and change `--image` accordingly; renaming an extension does not convert an image.

```bash
.venv/bin/python project.py predict-cracks --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt --image wall.jpg --output local-results/my-wall-v1 --device cpu
```

Wait for the command to finish. CPU runtime depends on the image and machine. It creates:

| File inside `local-results/my-wall-v1/` | Meaning |
|---|---|
| `predicted-crack-mask.png` | White pixels are proposed cracks; black pixels are proposed non-crack. |
| `probability.npz` | The numerical crack probabilities for further analysis. |
| `metadata.json` | Image/checkpoint hashes, dimensions, threshold and inference settings. |

Use the desktop file manager to open the mask and compare it with `wall.jpg` at the same size. This binary command does not produce a color overlay. For another run, choose a new output folder such as `local-results/my-wall-v2`.

The bundled checkpoint is a binary crack prototype. It can mistake joints or shadows for cracks and miss fine damage. An unlabelled photo provides no accuracy, Dice or IoU measurement. Four-class prediction and severity prediction require separately trained models; reviewed four-class masks, an approved severity rubric and independent evaluation are still needed. See [Usage](USAGE.md) and [Improving results](IMPROVING_RESULTS.md).

## 7. Return to the project later

For the ZIP method above:

```bash
cd "$HOME/Projects/heritage-damage-linux/heritage-damage-segmentation-public-main"
.venv/bin/python project.py status
```

For the Git method:

```bash
cd "$HOME/Projects/heritage-damage-segmentation-public"
.venv/bin/python project.py status
```

Reuse the existing environment; installation is not needed every session. If you move the project to another location or computer, create a new `.venv` there rather than copying the old environment.

## Troubleshooting

| Symptom | Action |
|---|---|
| `python3.12: command not found` or `ensurepip is not available` | Complete step 1, including `python3.12-venv`. Confirm the Ubuntu version before using its package names. |
| `externally-managed-environment` | Use `.venv/bin/python -m pip`, not the system `pip`; do not use `--break-system-packages`. |
| `No matching distribution found` for PyTorch | Check the Python version and `uname -m`, then use the official PyTorch selector for supported wheels. The commands here target `x86_64`. |
| `No module named ...` | Install requirements with the exact `.venv/bin/python` used to run the program. |
| `libGL.so.1` import error | Check `.venv/bin/python -m pip list` for an unintended GUI OpenCV package. A clean environment following step 3 uses `opencv-python-headless`; avoid mixing multiple OpenCV packages. |
| Download fails or checksum is not `OK` | Stop before extraction/loading. Check the public URL and retry the download; do not use an incomplete file. |
| Checkpoint or image missing | Run from the project root and check the exact path and letter case. Linux filenames are case-sensitive. |
| Output folder already exists | Choose a new run name; previous outputs are protected deliberately. |
| Process is killed or runs out of memory | Close other memory-heavy programs and start with a smaller development photo. Keep the original and record any resize as a changed input. |
| Dataset/rubric validation rejects training | Resolve missing reviews or approvals; do not change approval fields just to bypass validation. |

For all photo downloads and the limits of historical retraining, continue with [Assets](ASSETS.md). The [main installation guide](INSTALL.md) covers the other platforms.
