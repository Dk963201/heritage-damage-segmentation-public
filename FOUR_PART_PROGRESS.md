# Four-part implementation delivery — 18 October 2026

**Software is implemented and verified. The full research study remains incomplete.**

| Workstream | Delivered | Still required |
|---|---|---|
| Four-class segmentation | U-Net/DeepLabV3+/FPN training; native-resolution inference; four-class mask, probabilities and overlay; ROI-restricted pixel summaries | Approved four-class target masks and real target-domain experiments |
| Visual damage severity | Separate EfficientNet-B0 classification workflow; versioned rubric checks; reviewed-data validation; validation selection and test evaluation; prediction | the supervisor/authorized reviewer's actual rubric approval, reviewed severity labels and a trained/evaluated model |
| Erbil and Amedi | All800candidate files integrity-checked; both-site data gates; per-site evaluation; nine-photo pending review pack | Relevance, site/building identity, independent grouping, native-resolution labels and sufficient class/grade coverage |
| Deep CNN comparison | Reproducible three-architecture protocol, checkpoints, provenance and comparison export | Fair reviewed-data runs, frozen choices, independent final evaluation and revised research findings |

## Verified evidence

- **47 unit/regression tests passed**, zero failures, zero errors, zero skipped, in `local-results/four-part-verification-20261018-final/verification.json` and `tests.txt`.
- All three segmentation architectures completed a temporary synthetic training → checkpoint → evaluation → inference check. Temporary fixtures and checkpoints were removed. This demonstrates software execution only.
- Severity checks include a temporary synthetic CNN training/inference run, rejection of the pending rubric, source/group/city leakage checks, refusal to reuse original model-selection validation data as test, and rejection of unverified encoder initialization.
- All800archive members match source-manifest SHA256, bytes and dimensions. Those checks do not establish photographic relevance or damage labels.
- The three existing sparse user-reviewed binary masks are unchanged. Regions334–336 remain pending.
- No new approved heritage masks, severity labels, four-class research models or severity research models were created.

The current binary target-development score remains **69.23% crack Dice** on reviewed selections from three Erbil photographs. No80% achievement, Amedi accuracy, four-class accuracy or severity accuracy is claimed.

## Use the delivery

Read `ALL_FOUR_PARTS.md`, run `.venv-ml/bin/python project.py status`, and open `local-results/two-site-review-20261018/contact-sheet.jpg` for the next nine source photographs. The severity draft is deliberately pending. Relevance approval does not approve masks or severity labels.

This record describes the local implementation and verification before publication. Distribution files and current usage are indexed in docs/ASSETS.md and docs/README.md. Publication does not create new research results or approve pending labels. No teacher submission is part of this delivery.
