# Windows setup — PowerShell

This guide takes a new Windows user from downloading the project to a crack prediction. Use **64-bit Windows on an x86-64 PC, Python 3.12 and PowerShell**. Windows ARM64 needs its own compatible package selection. The commands were checked against the project and official documentation; **this project has not yet been executed on Windows**. The completed installation test was on [macOS](FRESH_INSTALL_CHECK.md).

Run each block in order. If a command fails, resolve it before continuing. An internet connection is needed for installation and downloads. You can start with one wall photo; the full 800-photo collection is optional.

## 1. Install Python 3.12

Install the **Python Install Manager** from [Python's Windows downloads](https://www.python.org/downloads/windows/). Open a new PowerShell window after installation. Then run:

```powershell
pymanager install 3.12
py -3.12 --version
py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"
```

The results should show `Python 3.12.x` and `64`. If `py -3.12` already works with a 64-bit installation, keep it and skip installing another copy. The distinct `pymanager` command avoids confusing the new installer with the older `py` launcher. See [Python's Windows instructions](https://docs.python.org/3/using/windows.html) if either command is missing.

## 2. Download and open the project

1. Open the [public repository](https://github.com/Dk963201/heritage-damage-segmentation-public).
2. Choose **Code → Download ZIP** and use **Extract All** in File Explorer.
3. Open the extracted folder containing `project.py` and `requirements-core.txt`. Do not run inside the compressed ZIP.
4. Right-click the folder background, choose **Open in Terminal**, and select a PowerShell tab.

Alternatively, with Git installed, run these commands from a folder where you keep projects:

```powershell
git clone https://github.com/Dk963201/heritage-damage-segmentation-public.git
Set-Location .\heritage-damage-segmentation-public
```

Choose one download route. If you chose the smaller pipeline ZIP instead, extract it into a new empty folder and start there. Use a short folder location to reduce Windows path-length issues.

Confirm you are in the correct folder:

```powershell
Test-Path .\project.py
Test-Path .\requirements-core.txt
```

Both should print `True`.

## 3. Create the environment and install dependencies

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```

These commands use CPU packages. The project's binary predictor has no CUDA device option, so an NVIDIA driver or CUDA installation is unnecessary for this workflow. Consult [PyTorch's installer](https://pytorch.org/get-started/locally/) if a compatible wheel is unavailable for your machine.

Always use `.\.venv\Scripts\python.exe` for this project. Activation and changes to PowerShell's execution policy are unnecessary.

## 4. Verify the installation

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch, segmentation_models_pytorch as smp, cv2, numpy; print('torch', torch.__version__, 'smp', smp.__version__)"
.\.venv\Scripts\python.exe project.py status
.\.venv\Scripts\python.exe -m unittest test_four_class test_metrics test_training_manifest
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_predict_four_class.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_project_entrypoint.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_severity_pipeline.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_two_site_review.py
```

Expect no broken dependencies and `OK` for each test command: **44 tests in total**. `INCOMPLETE` in the project status describes the research data still needed, rather than a failed installation.

An optional synthetic training/inference check exercises U-Net, DeepLabV3+ and FPN:

```powershell
.\.venv\Scripts\python.exe smoke_four_class.py
```

It writes `four-class-software-check.json`; run it in a fresh copy to preserve any previous evidence. Synthetic scores do not measure heritage accuracy.

## 5. Extract the trained binary model

The full source download already includes `assets/heritage-binary-model-20261018.zip`. If you downloaded the smaller pipeline package and this file is missing, get it with:

```powershell
if (-not (Test-Path .\assets\heritage-binary-model-20261018.zip)) {
    Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/Dk963201/heritage-damage-segmentation-public/main/assets/heritage-binary-model-20261018.zip" -OutFile .\assets\heritage-binary-model-20261018.zip
}
```

Check its identity, then extract into the current project folder:

```powershell
$modelHash = (Get-FileHash .\assets\heritage-binary-model-20261018.zip -Algorithm SHA256).Hash
if ($modelHash -ne "c5f1688bce81235bfb6b8dcec74fdeb8c40912e9723daf2848380f5d9181b3a8") {
    throw "Model archive checksum differs. Download the documented asset again before continuing."
}
Expand-Archive -LiteralPath .\assets\heritage-binary-model-20261018.zip -DestinationPath .
Test-Path .\local-results\reviewed-replay-long-v1\unet-reviewed-replay-long-final.pt
```

The final check should print `True`. On later runs, skip extraction if this checkpoint is already present. Do not force extraction over an existing experiment. [Microsoft documents ZIP extraction](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.archive/expand-archive) and [SHA-256 checking](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/get-filehash).

Verify the extracted checkpoint before loading it, including when reusing an existing file:

```powershell
$checkpointHash = (Get-FileHash .\local-results\reviewed-replay-long-v1\unet-reviewed-replay-long-final.pt -Algorithm SHA256).Hash
if ($checkpointHash -ne "757e6d8eac1f1d661dbbc7e698bdfc2890eca9388b7c6e66cd89e71797b02d21") {
    throw "Checkpoint checksum differs. Use a fresh project folder and the verified model archive."
}
```

## 6. Predict cracks in one photograph

Copy a clear wall photograph into the project folder and name it `wall.jpg`. To use a public example, download `heritage-versions-09.zip` from the [photo release](https://github.com/Dk963201/heritage-damage-segmentation-public/releases/tag/photos-v0.1-candidates), extract it separately, and copy `images/0650.jpg` as `wall.jpg`. Keep its `attribution/0650.json` credit record. Photo 0650 was part of training, so it demonstrates operation without providing an independent accuracy test.

```powershell
.\.venv\Scripts\python.exe project.py predict-cracks --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt --image wall.jpg --output local-results/my-wall-v1 --device cpu
Invoke-Item .\local-results\my-wall-v1\predicted-crack-mask.png
```

Outputs are in `local-results/my-wall-v1/`:

| File | Meaning |
|---|---|
| `predicted-crack-mask.png` | White pixels are predicted cracks; black pixels are predicted non-cracks. Compare with the original photo. |
| `probability.npz` | Per-pixel model probabilities for analysis. |
| `metadata.json` | Input/model hashes, dimensions, threshold and processing settings. |

For another photo, change `--image` and choose a new output name such as `local-results/my-wall-v2`. Quote paths containing spaces. Prediction alone produces no accuracy score; joints and shadows can be mistaken for cracks.

## 7. Return later or use VS Code

Open PowerShell in the same project folder and run the prediction command again with a new output name. You do not need to repeat installation or extraction. In VS Code, open this folder and use [Python: Select Interpreter](https://code.visualstudio.com/docs/python/environments#_select-an-environment) to select `.venv\Scripts\python.exe` if the Python extension is installed.

Record your installed versions before an experiment:

```powershell
.\.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding utf8 installed-environment.txt
```

## Troubleshooting

| Problem | What to do |
|---|---|
| `py` or `pymanager` is not recognized | Reopen PowerShell after installation and consult the official Python setup/troubleshooting page. If Python 3.12 is already installed, verify it before installing another copy. |
| PowerShell blocks `Activate.ps1` | Use `.\.venv\Scripts\python.exe` directly as above; activation is not needed. |
| Cannot find `project.py` or requirements | Open the extracted project folder, not its parent or the ZIP. The two `Test-Path` checks must succeed. |
| `No module named ...` | Install requirements with the same `.venv` interpreter that runs the program. |
| No matching wheel or DLL import failure | Check Python is 3.12/64-bit and use the official PyTorch selector for your architecture. Keep the full error for diagnosis. |
| Checkpoint missing after extraction | Keep the archive's `local-results` directory directly under the project root. |
| Output directory already exists | Use a new run name; existing predictions are protected. |
| Memory is exhausted | Close other memory-heavy programs and try a smaller input image, recording any resize as a separate demonstration. |

For the full 800-photo collection and historical reproduction limits, see [Assets](ASSETS.md). Four-class and severity training require additional reviewed labels and an approved rubric; see [Usage](USAGE.md) and [Improvement roadmap](IMPROVING_RESULTS.md).
