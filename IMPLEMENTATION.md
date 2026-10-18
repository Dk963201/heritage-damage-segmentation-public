# Assignment implementation status

| Stage | Evidence | Remaining requirement |
|---|---|---|
| Problem | Four-class task and local heritage objective documented | Confirm severity rubric and deadline |
| Collection | 800 candidate versions with hashes and provenance | Review relevance and annotate at least 800 usable originals |
| EDA | Candidate image audit and binary-source audit | Target pixel class balance and per-site coverage after masks |
| Models | Three required architectures in train_four_class.py; saved-checkpoint inference and exact protocol checks | Fair supervised comparison on approved target labels |
| Optimization | Focal plus Dice, Adam warmup, paired flips, color jitter, CLAHE | Train real target runs; repeated seeds/grouped validation |
| Diagnostics | Current three-photo Erbil binary development score69.23% crack Dice; per-site four-class metrics and tests | Held-out target confusion matrix, mIoU and class Dice |
| Severity | Tested image-level CNN workflow, pending draft rubric, reviewed-data and leakage checks | Actual rubric approval, reviewed grades and independent evaluation |
| Two-city review | All800archives verified; nine-photo pending review pack with source credits | Confirm building identity, relevance, class coverage and native-resolution masks |
| Deployment | Expert-assisted HCECR workflow proposed in report | Expert testing, severity labels and operational validation |

The binary pilot uses BCE plus Dice and AdamW in later experiments; it is a separate transfer-learning diagnostic, not evidence that the required four-class training recipe was executed. Synthetic end-to-end checks test software only. ALL_FOUR_PARTS.md is the current implementation guide; the earlier academic report remains a progress report, not a completed-study submission. No submission or teacher message has been sent.
