# Visual severity workflow

The software path is available. **Research training is blocked until the supervisor or another authorized reviewer supplies/approves a severity rubric and people assign reviewed image-level grades.** No real severity model or severity accuracy is supplied by this workflow. Existing binary crack masks do not provide severity labels.

`severity-rubric-draft.json` proposes the ordered grades **none, low, moderate, high** for discussion. It is deliberately `pending`, has no approver, and is rejected by training. The definitions concern visible surface appearance only. They do not establish structural safety, hidden damage, calibrated crack width or physical damaged area. There are no percentage-to-severity thresholds. Exclude unassessable photographs instead of calling them `none`.

## Prepare reviewed data

1. Obtain an approved rubric. Save a new version with `status: approved`, an actual `approved_by`, an `approved_on` date, released version, and agreed class definitions/examples. Do not merely change the draft status to bypass review. Class ordering must start at zero and match the list order.
2. Copy `severity-manifest-template.csv` into the directory containing the images. Add one row per independently reviewed image-level severity label. Paths are relative to that directory; paths outside it are rejected. Record a reviewer, review date, rubric ID/version, and the source file's SHA-256. `review_status` must be `reviewed`; pseudo-labels and model grades are refused.
3. Record verified site (`Erbil` or `Amedi`), building identity and near-duplicate/crop grouping. Keep each building and duplicate group in one split and one city. Groups must have real identifiers, not `unknown`, `pending` or `unassigned`. Every unrelated image still needs its own duplicate-group ID. Exact file duplicates and identical decoded RGB images are blocked throughout the dataset. Near duplicates, different views and crops still require human grouping; the software cannot prove their independence.
4. Assign nonempty `train`, `validation`, and `test` splits before training. Both sites must occur in **each split**, and training must cover every grade. Plan all grades in validation and test as well; missing classes are explicitly reported, not silently credited. Keep test images reserved until choices are frozen.

The validation checks confirm the recorded schema and hashes; they cannot certify that a named person actually reviewed labels or that building identities are correct. Review provenance remains a research responsibility.

## Commands

Run from this repository using the existing environment:

```bash
.venv-ml/bin/python severity_pipeline.py validate data/severity/manifest.csv \
  --rubric data/severity/rubric-approved-v1.json

.venv-ml/bin/python severity_pipeline.py train data/severity/manifest.csv \
  --rubric data/severity/rubric-approved-v1.json \
  --output local-results/severity-run-01 --epochs 20 --image-size 224 \
  --batch-size 4 --lr 0.0001 --seed 333 --random-init
```

Random initialization is the offline default; `--random-init` states it explicitly. Custom encoder initialization is deliberately unavailable until its training-source provenance can be verified against the severity validation/test groups. An existing binary model might already have seen those photographs, so loading it without that check would contaminate evaluation. The Python training API rejects any supplied initializer before creating output. No model or data downloads occur.

Architecture: SMP EfficientNet-B0 encoder, global average pooling and a linear classification head. Training uses ordinary cross entropy and AdamW on CPU, seeded data order and deterministic PyTorch algorithms. The whole RGB photograph is resized with preserved aspect ratio, centered on a black square, then ImageNet-normalized. No augmentation is applied. This small baseline may lose fine cracks during resizing; it is not an established severity model.

The earliest epoch with the highest **validation macro F1** is saved. Test evaluation is off by default. Outputs include the config, seeds, software versions, rubric, source hashes, dataset/manifest hashes, per-epoch validation results and selected model hash. Output directories must be new; prior experiments are never overwritten.

```bash
# Re-evaluate validation using the frozen checkpoint.
.venv-ml/bin/python severity_pipeline.py evaluate data/severity/manifest.csv \
  --rubric data/severity/rubric-approved-v1.json \
  --checkpoint local-results/severity-run-01/severity-best.pt \
  --output local-results/severity-validation-01

# Only after freezing model/rubric/experimental choices:
.venv-ml/bin/python severity_pipeline.py evaluate data/severity/manifest.csv \
  --rubric data/severity/rubric-approved-v1.json \
  --checkpoint local-results/severity-run-01/severity-best.pt \
  --split test --allow-test --output local-results/severity-test-01

.venv-ml/bin/python severity_pipeline.py predict \
  --checkpoint local-results/severity-run-01/severity-best.pt \
  --image path/to/photo.jpg --output local-results/severity-prediction-01
```

`train --evaluate-test` is also an explicit opt-in, but the separate test command makes the freeze easier to audit. Do not repeatedly use test results to tune. Evaluation checks the model's training image/building/duplicate identifiers against the chosen evaluation rows, including when a different manifest is supplied. Test evaluation also excludes the original validation images/groups used to select the epoch. A checkpoint without that validation provenance is refused for test evaluation. Re-evaluating the original validation set remains allowed. A missing severity checkpoint produces an unavailable error; the binary segmentation model is not silently substituted.

## Interpreting outputs

Metrics include confusion matrix (rows true, columns predicted), support, precision, recall, F1, macro F1, accuracy, and mean absolute ordinal error in **class steps**, plus separate Erbil/Amedi slices. Precision/recall are `null` when their denominators are zero. F1 is `null` when a class is absent from both truth and predictions; it is zero when a present or predicted class has no true positives. Macro F1 averages defined class F1 values and records how many classes contributed and which true classes were absent. Empty site slices have `null` aggregate metrics. Missing coverage is not evidence of good performance.

Prediction JSON preserves class order and provides **raw, uncalibrated softmax probabilities** and a proposed visible-severity grade. These are not confidence guarantees, reviewed labels, physical measurements or structural safety decisions. Inference without a reference grade reports no accuracy. Checkpoints use `weights_only=True` loading and local, fixed architecture construction.

## Software verification

```bash
.venv-ml/bin/python -m unittest discover -s tests -p 'test_severity_pipeline.py' -v
```

Fixtures are explicitly synthetic and temporary. Tests cover pending/unnamed rubric refusal, label review/hash/site validation, duplicate/building leakage, missing splits, absent-class/empty-site metrics, ordinal order, missing models, deterministic inference, overwrite refusal, safe checkpoint consistency and a small training/evaluation smoke run. Synthetic metrics are **software test outputs only**, never findings about heritage photographs.

Python exports for integration: `validate_rubric(rubric_dict)`, `validate_dataset(manifest_path, rubric_dict)`, `classification_metrics(truth_indices, predicted_indices, class_order)`, `train(...)`, `evaluate(...)`, `predict(...)`.
