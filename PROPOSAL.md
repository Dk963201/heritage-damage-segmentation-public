# Project proposal

**Acquisition update,18 October 2026:**800 unannotated candidate image versions (609 Erbil,191 Amedi) are now downloaded and integrity-verified. Visual selection, near-duplicate review and pixel masks remain pending. Earlier initial-discussion notes below describe the plan before this acquisition. See IMAGE_DOWNLOADS.md.

## Objective

Develop and compare semantic segmentation models that identify healthy masonry, visible cracks, surface erosion and vegetation overgrowth in photographs of heritage structures in Erbil and Amedi. Produce labelled overlays and class-area summaries for expert review. A photograph alone cannot establish structural safety; predicted damage area is not an engineering severity rating.

## Research question

How do U-Net with an EfficientNet encoder, DeepLabV3+ and FPN compare on mean IoU and per-class Dice under a shared, leakage-controlled evaluation protocol, particularly for thin cracks and rare erosion regions?

## Initial deliverable

The project title is selected. A licensed 100-image UK masonry crack dataset is selected as a **binary pilot only** to test pairing, annotation inspection and the pipeline. It cannot establish performance on Kurdish citadels and cannot satisfy the four-class, 800-image assignment requirement. See DATASET.md for the acquisition gap and source details.

## Proposed study design

1. Obtain permission and collect at least 800 original high-resolution photographs of relevant Kurdish heritage masonry; confirm whether the requirement counts originals rather than patches.
2. Record the structure/wall/session for each image. Keep related images and all derived patches in the same partition. Aim for a held-out structure and, if the number of independent structures allows, a held-out site evaluation.
3. Annotate four classes with an additional ignored region for sky, people, unrelated objects and uncertain boundaries. Do not label every unmarked pixel as healthy masonry.
4. Inspect blur, exposure, duplicate frames, label agreement and class distribution before splitting. Avoid claiming that 800 similar patches are 800 independent images.
5. Compare the three requested architectures on identical data. Tune only with training/validation partitions and evaluate the frozen test set once.
6. Report class-wise IoU/Dice, mean IoU over supported classes, pixel accuracy and errors by site/lighting. Include a constant-class baseline and limitations.

Numerical performance targets will be set after data inspection, not invented in advance. Patch size, resolution threshold and severity labels require agreement with the instructor; missing dimensions in the source extraction are not inferred.

## Short explanation for the supervisor

“My selected topic is semantic segmentation of damage in Erbil and Amedi heritage masonry. I will compare U-Net/EfficientNet, DeepLabV3+ and FPN across four labelled classes. I have prepared the project structure and identified a licensed crack-mask pilot. The pilot is UK data with only crack labels, so I still need approval and access to collect the required Kurdish four-class dataset. I will split by structure before making patches to avoid data leakage.”

## Questions to resolve today

- Is the public binary pilot acceptable while local imagery is collected?
- Can the department provide imagery, site contacts or annotation support for the 800-image requirement?
- Are the 800 images original photographs, and what minimum resolution is expected?
- What is the final deadline, and are separate severity labels required beyond segmentation?
- What rubric and permitted assistance/disclosure rules apply to the final report and code?

No collection permissions, dataset approval or marking outcome are claimed.
