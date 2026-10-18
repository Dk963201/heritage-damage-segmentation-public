# Download and install

The core program is Python. Use Python 3.12 for a close match to development (macOS ARM64, Python 3.12.14). Windows and Linux instructions below are provided, but those operating systems were not executed in this delivery. CPU is the common supported route; the current segmentation and severity trainers run on CPU. The binary predictor additionally supports Apple MPS. Installing CUDA does not automatically make these scripts use an NVIDIA GPU.

For complete instructions from installing Python to opening your first prediction, choose the [Windows PowerShell guide](INSTALL_WINDOWS.md) or the [Linux guide (Ubuntu 24.04 example)](INSTALL_LINUX.md). The quick setup commands below also cover macOS.

## 1. Download the project

Open the public [repository](https://github.com/Dk963201/heritage-damage-segmentation-public), choose **Code → Download ZIP**, and extract it. Open a terminal in the extracted `heritage-damage-segmentation-public-main` directory. Alternatively, if Git is installed:

```sh
git clone https://github.com/Dk963201/heritage-damage-segmentation-public.git
cd heritage-damage-segmentation-public
```

For the self-contained delivery, extract `heritage-four-part-pipeline-20261018.zip` into a **new empty folder**. That archive preserves nested tests, review thumbnails and evidence folders. Do not overlay it onto existing annotations. The image dataset and trained binary weights are separate downloads; see [Assets](ASSETS.md).

## 2. Create an environment

Install Python from [python.org](https://www.python.org/downloads/) or your OS package manager. Run one section below from the project directory.

### Windows — PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe project.py status
```

You do not need to change PowerShell's execution policy or activate the environment. In every later command, replace `python` with `.\.venv\Scripts\python.exe`.

### macOS — Terminal

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python project.py status
```

In later commands replace `python` with `.venv/bin/python`. Apple Silicon is the tested hardware family. On an older Intel Mac, choose a compatible wheel/Python combination from the official PyTorch guidance; do not assume the newest wheel supports every Mac.

### Linux — terminal

Install Python 3.12 with its `venv` module using your distribution's package manager, then:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-core.txt
.venv/bin/python project.py status
```

In later commands replace `python` with `.venv/bin/python`.

## 3. Check the installation

```sh
python -m pip check
python -c "import torch, segmentation_models_pytorch as smp, cv2, numpy; print('torch', torch.__version__, 'smp', smp.__version__, 'CUDA', torch.cuda.is_available(), 'MPS', torch.backends.mps.is_available())"
python -m unittest test_four_class test_metrics test_training_manifest
python -m unittest discover -s tests -p test_predict_four_class.py
python -m unittest discover -s tests -p test_project_entrypoint.py
python -m unittest discover -s tests -p test_severity_pipeline.py
python -m unittest discover -s tests -p test_two_site_review.py
```

These select the 44 self-contained tests and require no photo/model download. The recorded 47-test local run additionally included three real binary-checkpoint integration tests. Optional `python smoke_four_class.py` runs synthetic training/inference through all three architectures; its scores are not heritage results. It writes `four-class-software-check.json`, so run it in a fresh copy if preserving that evidence file matters.

`requirements-core.txt` is a small portable dependency specification. The environment lock files (`requirements-*-lock.txt`) record the actual local environments, including transitive packages; they are provenance, not proof that every pinned wheel is available on every OS. Freeze your successful installation with `python -m pip freeze > installed-environment.txt` and record OS/hardware before an experiment. Different packages/devices may change numerical results.

The [fresh-download verification](FRESH_INSTALL_CHECK.md) records a clean-environment run of these steps, the corrected optional smoke test and a real-photo prediction.

## Common problems

| Problem | Action |
|---|---|
| GitHub shows 404 | Check the public repository URL and release name; old private-copy links are not used by this distribution. |
| `No module named ...` | Run pip with the same environment Python used to run the program. |
| No compatible PyTorch wheel | Use the official install selector for your OS, Python and hardware. Do not silently install an unrelated package. |
| Missing checkpoint/photo | Download and extract the specific asset from [Assets](ASSETS.md); code alone is insufficient. |
| Output folder already exists | Choose a new run name. The program deliberately protects previous outputs. |
| Training rejected the manifest/rubric | Resolve the stated data-review problem. Do not mark labels or the severity rubric approved just to bypass a check. |
| Large photos exhaust memory | Start with a small development example and CPU; preserve originals and record any resize/crop as a new protocol. Full 800-photo training can be expensive. |

Official installation references: [Python virtual environments](https://docs.python.org/3.12/library/venv.html), [PyTorch platform selector](https://pytorch.org/get-started/locally/), [GitHub source downloads](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives). Check those sources for current OS/wheel support.
