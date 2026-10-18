# Data quality and accuracy improvement plan

Current target pool: 800 Wikimedia Commons candidate images (609 Erbil, 191 Amedi), stored as rendered 3,840-pixel versions. This is an acquisition pool, **not 800 damage examples or labeled samples**. All 800 require visual screening; zero are accepted four-class pixel masks. Four records marked Public domain have no direct license URL and need source-page review. The originals/source URLs, authors, license labels and SHA-256 values are retained in `downloaded-800-manifest.csv`.

## Ready-to-review files

- `screening-manifest.csv`: one row per candidate. Complete provenance review, visual suitability, physical `structure_id`, capture group, duplicate group and visible-class fields. Do not assign splits until related views/crops/duplicates are grouped.
- `annotation-queue.csv` and `ANNOTATION_PILOT.md`: prior 37-image thumbnail triage. This is not independent review and is not mask approval; it must be reconciled with the new manifest before any split.
- `Dataset.md`: class conventions, including 255 ignore for background/uncertain pixels. Confirm the project's treatment of mortar, exposed masonry and structural significance with the supervisor before annotation.

## Sequence

1. Resolve source-page license/provenance checks. Retain attribution and license per image. Exclude unclear permission from training and publication until resolved.
2. Human-screen original images for visible masonry, useful scale/focus, image artifacts, whether damage is actually visible, and whether the scene is relevant. Mark unsuitable/context-only separately; never count as annotated examples.
3. Assign building/structure and capture-session IDs; detect exact and near duplicates. Keep all views/crops from one structure together.
4. Agree written four-class rules with the supervisor or a conservation-domain reviewer, especially the term “structural crack”: images alone cannot prove structural significance. Use “visible crack” unless expert evidence supports the stronger claim. Use ignore for uncertain pixels.
5. Pilot double-annotation on 20–30 suitable images, measure class-wise IoU/agreement and resolve disagreements. Only then scale annotation in CVAT and record annotator/reviewer/version.
6. After screening and grouping, freeze a structure-level split (planning start: 70/15/15 where enough independent structures exist). Split before crops/augmentation. Hold out buildings for validation/test; reserve a final untouched test set and report per-class Dice/IoU and confidence intervals.
7. Train controlled baselines on reviewed masks only. Compare resolution, augmentation, loss and model on validation. Never tune on the final test set. Report binary crack experiments separately from Kurdish four-class results.

## Current blockers

The candidate pool may contain many scenic/context images without visible target damage. It has no verified structure IDs, no masks and no approved severity rubric. Therefore no honest four-class accuracy claim or 80% target is currently measurable. Human visual screening, attribution/licensing check, expert class rules and independent building groups are prerequisites; pseudo-labels remain drafts.
