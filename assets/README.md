# Downloadable research assets

Start with [the asset guide](../docs/ASSETS.md) for download, extraction, checksums, sources and limitations.

- [Binary crack model](heritage-binary-model-20261018.zip): final U-Net/EfficientNet-B0 weights and experiment context. Extract into the project root, retaining the `local-results/` paths.
- [Reviewed labels and evidence](reviewed-labels-evidence-20261018.zip): sparse reviewed labels through region 333, review records and selected evidence. Regions 334–336 remain pending.
- [Asset catalog](asset-catalog.json) and [checksums](SHA256SUMS.txt): exact package contents and identity.
- [Pipeline code/review package](../heritage-four-part-pipeline-20261018.zip): software, tests, handbook and pending nine-photo two-site review pack.
- [800-photo release](https://github.com/Dk963201/heritage-damage-segmentation-public/releases/tag/photos-v0.1-candidates): separately downloaded candidate photographs with source credits.

No trained target four-class or severity model is included. The final binary model has no independent test score; the 69.23% Dice figure belongs to reused three-photo development folds. Third-party data licenses remain with their source owners.
