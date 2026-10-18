# Fresh public download check

On **18 October 2026**, the public repository was downloaded anonymously into an empty folder and installed in a new Python 3.12 virtual environment. No existing project environment, photo, model cache or downloaded checkpoint was reused. The published binary-model ZIP and photo archive 09 were downloaded separately and their checksums verified.

## Results

| Check | Result |
|---|---|
| Documented macOS installation | Passed: new environment, PyTorch, torchvision and `requirements-core.txt` |
| `python -m pip check` | No broken requirements found |
| `project.py status` and `--help` | Passed; research status correctly remains incomplete |
| Documented self-contained tests | **44 passed** |
| Synthetic end-to-end check | **U-Net, DeepLabV3+ and FPN passed** training, checkpoint reload, evaluation and inference |
| Binary prediction using the published model | Passed on full-size candidate photo 0650: **3840 × 2563 pixels, 48 tiles, CPU, 23.38 seconds** |
| Output integrity | Correct dimensions, binary mask values, finite probabilities, matching threshold and file hashes |

The initial public source was commit `53d81fe956581998972abc0ee716766cd7961110`. Its optional smoke test failed because it expected a development environment named `.venv-ml`. The fix uses `sys.executable`, so child commands use the same Python interpreter that started the test. After applying that fix to the downloaded copy, the complete synthetic check passed in 10.79 seconds. The current source and pipeline download include this fix.

Linked guides were also corrected to use the documented environment on each operating system. The asset guide now states the extra annotation-workspace files needed to rebuild the historical two-site review pack; those files are not supplied by the public photo archives. The supplied review page can be opened directly.

## Environment and reproduction

Tested on **macOS ARM64**, Python **3.12.14**, PyTorch **2.14.1**, torchvision **0.29.1**, segmentation-models-pytorch **0.5.0**, NumPy **2.5.4**, OpenCV **4.14.0** and Pillow **13.0.0**. Package resolution used the public Python package index with no pip cache or extra package index. Tests ran with empty model caches and offline pretrained-weight access.

Follow [Installation](INSTALL.md) for the environment and 44 tests, then run the optional synthetic check with the same environment Python:

```sh
python smoke_four_class.py
```

For the real-photo example, download [the model and photo archive 09](ASSETS.md), extract the model into the project directory, and copy `images/0650.jpg` from archive 09 to `wall.jpg`. Keep its matching attribution record. Then run:

```sh
python project.py predict-cracks --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt --image wall.jpg --output local-results/my-wall-v1 --device cpu
```

As in the installation guide, replace `python` with `.venv/bin/python` on macOS/Linux or `.\.venv\Scripts\python.exe` on Windows. Use a new output directory for each run.

The test produced `predicted-crack-mask.png`, `probability.npz` and `metadata.json`. White mask pixels represent proposed cracks. [Machine-readable results](fresh-install-check.json) preserve hashes and measured timings; [the resolved dependency record](requirements-fresh-install-lock.txt) records this environment without asserting portability to every platform.

## What this establishes

This checks that a fresh user can install the software, run its tests and produce a prediction from public downloads. **Windows and Linux were not executed.** Timings depend on the machine.

Photo 0650 was part of the reviewed training data. This demonstration therefore measures **no independent accuracy, Dice or IoU** and does not improve the reported research score. Synthetic checks likewise do not establish heritage accuracy. Target four-class training, severity training and a fair independent evaluation still require the reviewed data and rubric described in [Project scope](PROJECT_SCOPE.md) and [Improvement roadmap](IMPROVING_RESULTS.md).
