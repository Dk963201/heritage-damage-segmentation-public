# The four-part heritage study

Full topic: **Semantic Segmentation and Damage Severity Classification of Historical Citadel Structures in Erbil and Amedi using Deep Convolutional Neural Networks**.

This is a working research pipeline with a trained binary crack prototype. The full research study is **not complete**. Four-class and severity software can be tested before reviewed data exists, but those tests are not model-performance evidence.

## What each part does

| Part | Software path | Evidence needed for research completion |
|---|---|---|
| Semantic segmentation | `train_four_class.py` and `predict_four_class.py`: four pixel classes, ignored/background pixels, aligned overlay and class-pixel summaries | Reviewed masks for healthy masonry, visible cracks, erosion and vegetation; held-out per-class Dice/IoU and mean IoU |
| Damage severity | `severity_pipeline.py`: a separate image-level CNN classifier with an approved rubric, reviewed labels, training, evaluation and prediction | A defined visual-severity outcome, qualified label review, independent test labels, confusion matrix, macro F1 and ordinal error |
| Erbil and Amedi | Candidate inventory and a pending two-site review packet; site-aware validation and evaluation | Confirmed site/building identity, usable annotated photographs and independent buildings in every split; adequate class coverage in each city |
| Deep CNN comparison | U-Net, DeepLabV3+ and FPN with EfficientNet-B0 encoders and the same four-class experiment protocol | Train on the reviewed target dataset, compare validation results fairly, freeze choices, then report the held-out test |

The existing binary model is the crack-detection foundation. Its 69.23% pooled crack Dice is development evidence from sparse labels on three Erbil photographs. It is not an Amedi result, a four-class score, severity accuracy or an independent final test. More regions on the same photograph do not increase the number of independent buildings.

## One entry point

Run commands from this project directory using the existing environment:

```sh
.venv-ml/bin/python project.py status
.venv-ml/bin/python project.py check-segmentation-data training-manifest.csv
.venv-ml/bin/python project.py train-segmentation training-manifest.csv --output local-results/four-class-target-v1 --epochs 30 --warmup 5
```

The last two commands require a real manifest: copying the empty template is not sufficient. The entry point requires Erbil and Amedi in **each** train/validation/test split, reviewed masks, all four classes in training, verified source hashes, no duplicate sources, and no building-group overlap. These are minimum checks, not proof of adequate sample size or correct human labels. Confirm all class supports and review near-duplicates before training.

After training, inspect `comparison.json`, each architecture's validation history, source hashes and protocol. Test evaluation remains off until choices are frozen; `--evaluate-test` opts in. Site metrics are available under `by_site`; missing classes have undefined scores, not perfect scores. Do not choose hyperparameters by repeatedly reading the test score.

Four-class prediction after a trained checkpoint exists:

```sh
.venv-ml/bin/python project.py predict-segmentation --checkpoint local-results/four-class-target-v1/unet-best.pt --protocol local-results/four-class-target-v1/protocol.json --model unet --image wall.jpg --roi wall-roi.png --output local-results/wall-four-class-v1
```

The ROI must match the photograph dimensions and select the masonry to analyze. Without a supplied ROI, whole-image class fractions include background and are explicitly unsuitable for estimating wall damage. Read `FOUR_CLASS_INFERENCE.md` for the exact ROI format and output contract. Pixel fractions are not calibrated physical areas or severity grades.

Severity commands, once an approved rubric and reviewed labels exist:

```sh
.venv-ml/bin/python project.py check-severity-data severity-manifest.csv --rubric severity-rubric-approved.json
.venv-ml/bin/python project.py train-severity severity-manifest.csv --rubric severity-rubric-approved.json --output local-results/severity-target-v1 --epochs 30
```

Read `SEVERITY_WORKFLOW.md` for prediction commands and model artifacts. `severity-rubric-draft.json` is a proposal for discussion, not approval. The user confirmed on 18 October 2026 that the lecturer has not supplied a rubric. Do not change its status to approved without a documented review decision. This version predicts visual damage categories only; it cannot determine structural safety from a photograph.

The existing crack predictor remains available independently:

```sh
.venv-ml/bin/python project.py predict-cracks --checkpoint local-results/reviewed-replay-long-v1/unet-reviewed-replay-long-final.pt --image wall.jpg --output local-results/wall-binary-v1
```

All output directories must be new so existing experiments and annotations are preserved. The selected binary weights are distributed separately from source code; see docs/ASSETS.md. No trained four-class or severity weights exist yet.

## What the user can review next

Open the nine-photo contact sheet in `local-results/two-site-review-20261018/` and consult `TWO_SITE_DATA_REVIEW.md`. These are **pending source photographs**, not accepted masks. Decide which photographs contain sufficiently clear masonry for pixel annotation; identify the building where known, and mark uncertain cases pending. Related views must stay together in one building group.

For selected photographs, mark all four visible classes with a consistent boundary policy. Ignore sky, people, objects outside the task and ambiguous surfaces; do not label every unlabeled pixel healthy. Existing binary crack selections are retained as a separate pilot rather than silently converted into complete four-class masks.

Review the proposed severity definitions with the lecturer or an appropriate conservation specialist before assigning severity outcomes. Save the final rubric version, named reviewer and independent labels. The software cannot create evidence of human review.

## Heritage context

Erbil Citadel has been inscribed on the UNESCO World Heritage List since 2014: https://whc.unesco.org/en/list/1437. Amedy city is on Iraq's Tentative List, submitted in 2011: https://whc.unesco.org/en/tentativelists/5601/. Both are relevant to heritage preservation, but their UNESCO status differs. The candidate images came from Wikimedia Commons with individual source credits, not from a UNESCO-labeled machine-learning dataset. No institutional partnership or deployment is claimed.

## Completion evidence still required

The documented assignment scope calls for 800 suitable high-resolution source images. Confirm with the lecturer whether downloaded rendered versions are acceptable; 800 candidate versions are not proof of 800 usable independently annotated photographs. Prepare the reviewed target datasets, audit class/grade balance and building groups, run and freeze the experiments, evaluate held-out images from both sites, then revise the research report with actual findings and limitations. No 80% performance guarantee is made.
