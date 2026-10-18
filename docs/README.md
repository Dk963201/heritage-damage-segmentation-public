# Heritage project handbook

Read these pages in order. Commands run from the repository root.

1. [Purpose and four project parts](PROJECT_SCOPE.md): what we want to accomplish and what is still missing.
2. [Assets and downloads](ASSETS.md): photographs, labels, trained binary model, software, evidence and their licenses.
3. [Install on Windows, macOS or Linux](INSTALL.md): download, environment, dependencies and checks.
4. [Run predictions and training](USAGE.md): runnable commands, inputs and outputs.
5. [Method, equations and prediction process](METHOD_AND_PREDICTION.md): how the model learns and how we measure it.
6. [What must happen for higher results](IMPROVING_RESULTS.md): data priorities, fair evaluation and a measurable improvement plan.
7. [Fresh-download verification](FRESH_INSTALL_CHECK.md): installation, tests, prediction evidence and platform limits.

New to installation? Follow the complete [Windows PowerShell guide](INSTALL_WINDOWS.md) or [Linux guide](INSTALL_LINUX.md), including model extraction, a first prediction and troubleshooting. macOS commands remain in [Install](INSTALL.md).

## Repository organization

- `docs/` is the current user handbook.
- `assets/` holds packaged model/review assets and their inventory.
- Root Python files are the maintained commands. Their locations are retained so recorded experiments and imports continue to work.
- `tests/` plus root `test_*.py` files verify the software.
- `local-results/` is the local experiment workspace. The downloadable pipeline archive contains the selected evidence and two-site review folders needed to inspect this delivery.
- Candidate photographs are separate release assets; a source ZIP alone is not the image dataset.
- Older root reports remain the experiment history. In particular, `ACADEMIC_REPORT_HISTORICAL_PUBLIC.pdf` and `.docx` predate the reviewed-label experiment and are progress reports, not the final research paper.

The current handbook supersedes old quick-start instructions. Historical measurements remain tied to their original datasets and must not be compared as if they were one test.
