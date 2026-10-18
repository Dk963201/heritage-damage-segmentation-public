# Local run results — 18 October 2026

**All 800 images passed the integrity and CLAHE preprocessing run. No supervised model has been trained.**

| Check | Actual result |
|---|---:|
| Images processed | 800/800 |
| Processing/hash/dimension failures | 0 |
| Erbil / Amedi | 609 / 191 |
| Unique byte hashes | 800 |
| Potential 256×256 windows | 141,122 |
| Similarity pairs requiring manual review | 6 |
| Full-image run time | 155.64 seconds |
| Pixel-annotated images | 0 |
| Pixel accuracy / mean IoU / class Dice | Not available — labels required |

Completed at 2026-10-18T07:38:47.144747+00:00 (10:38 Baghdad). Patch counts are calculated, not extracted training examples; no model was trained on these windows. The preprocessing run discarded processed arrays after measuring them to conserve disk space.

All seven unit tests passed. U-Net, DeepLabV3+ and FPN, each with EfficientNet-B0, passed synthetic forward/backward and Adam optimizer-step checks. These random-input tests establish software compatibility only and contain no heritage-image evaluation.

## Evidence and reproduction

- [Per-image audit](image-audit.csv)
- [Machine-readable run summary](run-summary.json)
- [Similarity review pairs](similarity-review.csv)
- [Synthetic model checks](model-smoke-results.json)
- [Unit tests](test-results.txt)
- [Exact environment](requirements-ml-lock.txt)
- [Process and VS Code instructions](LOCAL_RUN.md)
- [Annotation review template](annotation-review-template.csv)

Next: visual relevance review, expert-checked pixel annotations and leakage-safe grouped splits. The full supervised trainer and real held-out experiment remain to be implemented after these data requirements are resolved. No research accuracy or damage severity has been inferred from these processing checks.
