# Heritage Damage Study

**Semantic Segmentation and Damage Severity Classification of Historical Citadel Structures in Erbil and Amedi using Deep Convolutional Neural Networks**

We want to locate visible cracks, erosion and vegetation in heritage-building photographs, compare three CNN architectures, and eventually predict an independently reviewed visual-severity grade. The practical aim is to help conservation reviewers organize photographic inspections and identify surfaces needing closer attention.

**Today:** a trained binary crack prototype and tested four-class/severity software are available. The complete research study still needs reviewed four-class data from both cities, an approved severity rubric and independent evaluation. No structural-safety decision or completed four-class result is claimed.

This is the [public research snapshot](PUBLIC_RELEASE.md), with fresh history and personal student details removed.

## Start here

| I want to… | Open |
|---|---|
| Understand the goal, four project parts and four pixel classes | [Project scope](docs/PROJECT_SCOPE.md) |
| Find the images, labels, model and evidence | [Assets and downloads](docs/ASSETS.md) |
| Download and install on Windows, macOS or Linux | [Installation](docs/INSTALL.md) |
| Run a prediction or prepare training | [Step-by-step usage](docs/USAGE.md) |
| Explain our algorithms, parameters and equations | [Method and prediction](docs/METHOD_AND_PREDICTION.md) |
| Know what we must do for higher results | [Improvement roadmap](docs/IMPROVING_RESULTS.md) |
| See the fresh-download test results | [Installation verification](docs/FRESH_INSTALL_CHECK.md) |
| Browse the organized handbook | [Documentation index](docs/README.md) |

## What the four parts mean

| Project part | Implemented | Evidence still needed |
|---|---|---|
| Semantic segmentation | Pixel classes: healthy masonry, visible crack, erosion, vegetation; masks, probabilities and overlays | Reviewed target masks, trained target models and held-out class metrics |
| Visual severity | Separate EfficientNet-B0 image classifier with rubric and label checks | Approved definitions, reviewed grades and real training/evaluation |
| Erbil and Amedi | 800 candidate-image inventory and a pending two-site review pack | Verified buildings and usable independent examples of every relevant class |
| Deep CNN comparison | U-Net, DeepLabV3+ and FPN under one training/evaluation protocol | Fair experiments on reviewed target data and a frozen independent test |

Four project parts are not four severity grades. The pending severity definitions are a proposal, not the lecturer's approved rubric.

## Current measured result

| Metric | Binary crack development result |
|---|---:|
| Crack Dice | **69.23%** |
| Crack IoU | 52.94% |
| Precision | 95.94% |
| Recall | 54.16% |

These results use 177,493 selected reviewed pixels from **three Erbil photos**, through region 333. Repeated photo-holdout experiments make this development evidence; there is no untouched test or Amedi score. The final downloadable model was trained on all three photos and has no independent test score. **80% crack Dice has not been reached.** See [all experiments, including failures](REVIEWED_IMPROVEMENT_RESULTS.md). The older 73.25% UK-masonry score used a different dataset and cannot be treated as a higher Kurdish result.

## Download and try it

1. Download the repository source or [the self-contained pipeline package](heritage-four-part-pipeline-20261018.zip).
2. Follow [your OS installation steps](docs/INSTALL.md).
3. Download and extract [the binary crack model](assets/heritage-binary-model-20261018.zip) into the project root.
4. Run the following with your environment's Python and a local `wall.jpg`:

```sh
python project.py status
python project.py predict-cracks --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt --image wall.jpg --output local-results/my-wall-v1 --device cpu
```

White pixels in `predicted-crack-mask.png` are model-proposed cracks. Compare them with the original wall photograph; they are not new approved labels. For the full dataset, use the [800-candidate photo release](https://github.com/Dk963201/heritage-damage-segmentation-public/releases/tag/photos-v0.1-candidates). Code, photos, labels and weights are different assets; [the asset guide](docs/ASSETS.md) explains each.

## What must happen next

Review additional buildings from both cities, annotate complete relevant surfaces and difficult joints/shadows, agree severity definitions with the supervisor, reserve whole buildings for a locked test, then train and compare models. More regions from one existing photo do not establish generalization. Follow the [prioritized improvement plan](docs/IMPROVING_RESULTS.md).

The four-part implementation passed **47 local software tests**, including three existing binary-checkpoint tests. The portable package includes **44 self-contained tests** and the recorded verification evidence. Synthetic checks are not model-accuracy results. Windows/Linux setup is documented but not executed; the local run used macOS ARM64/Python 3.12.

## Repository map

- `docs/`: current purpose, assets, installation, usage, methods and improvement handbook.
- `assets/`: downloadable binary weights, reviewed-label/evidence pack and checksums.
- `project.py`: common command entry point; other root Python files implement training, inference and validation.
- `tests/` and root `test_*.py`: software verification.
- `local-results/`: local experiment outputs; the pipeline ZIP preserves selected evidence and pending review folders.
- Older root guides and Word/PDF reports: preserved experimental history. The existing academic report predates the reviewed-label experiment and is not the final completed paper.

The 800 candidates are individually credited image versions, primarily from Wikimedia Commons, not an official UNESCO-labeled dataset. Keep author/license records with images. No institutional partnership, validated structural assessment or performance guarantee is claimed.
