# Completing the four parts of the heritage study

## Iteration contract — 18 October 2026

Goal: finish the reproducible software paths for four-class segmentation, visual-damage severity classification, two-site dataset preparation, and a fair CNN comparison. Research completion additionally requires reviewed data and independent evaluation; software tests cannot substitute for these.

Intended use: help a conservation reviewer locate visible damage and organize inspection priorities. A photograph does not establish structural safety, calibrated crack width, or physical damaged area.

Data snapshot: 800 candidate image versions (609 Erbil, 191 Amedi), with sparse user-reviewed binary selections on three Erbil photographs. No approved four-class training manifest or severity-label dataset is presently available. Existing labels, predictions and completed reviews must remain unchanged.

Baseline: the current binary U-Net/EfficientNet-B0 development result is 69.2336% pooled crack Dice. It uses repeatedly examined three-photo development folds, not an independent final test. The 80% aspiration remains unmet and is not a guarantee.

Success evidence for this iteration:

- Runnable four-class inference matching the existing training preprocessing and checkpoint format, with per-class masks, overlays and descriptive pixel summaries.
- A severity CNN training/evaluation/inference path that requires a versioned, approved visual-severity rubric and independently reviewed labels. No invented percentage-to-severity thresholds.
- An auditable inventory and a pending review pack covering both sites, without fabricating building identity, class labels or review approval.
- One clear entry point, reproducible commands, meaningful validation tests, and an explicit readiness report separating software readiness from research evidence.

Research metrics: per-class Dice/IoU and mean IoU for segmentation; confusion matrix, per-class recall/F1, macro F1 and ordinal error for severity when the rubric defines ordered classes. Report both site slices. Keep photographs, crops and near duplicates from the same building in one split; tune on validation only and reserve test data until choices are frozen.

Unacceptable mistakes: treating pseudo-labels as reviewed truth, assigning severity from uncalibrated area alone, claiming Amedi performance without Amedi evaluation, reusing training imagery as independent evidence, overwriting user labels, or reporting synthetic test scores as scientific findings.

Open decision: the user confirmed on 18 October 2026 that the supervisor has not provided a severity rubric. A pending draft is prepared for review; research severity training rejects it until an actual approval decision and reviewed labels exist.

Next experiment after data review: train the same four-class protocol with U-Net, DeepLabV3+ and FPN on a building-group split containing both sites, then evaluate severity against independently assigned labels. Freeze the protocol before the final test.

Fallback: preserve the existing binary model and identify its output explicitly as an experimental crack prediction. Missing four-class or severity models should produce a clear unavailable status, never a fabricated grade.
