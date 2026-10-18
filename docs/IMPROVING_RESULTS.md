# What we must do to improve results

**The next priority is better, more diverse reviewed data from both cities—not another promise that more training will reach 80%.** The software can train and evaluate models, but it cannot manufacture correct labels, independent buildings or an approved severity rubric.

Run the commands below from the repository root, using `python` in the project's activated Python environment. They work the same way on Windows, macOS and Linux once the dependencies and required local files are installed. Paths for future reviewed datasets are examples; those datasets do not already exist merely because a command is documented.

## What the current result means

| Evidence | Current state |
|---|---|
| Best reviewed-target binary crack Dice | **69.2336%**, selected from reused development experiments |
| Precision / recall / IoU | **95.94% / 54.16% / 52.94%** |
| Target labels evaluated | 177,493 selected pixels from **three Erbil photographs** |
| Independent buildings or Amedi test performance | Not established |
| Approved four-class target dataset | Not yet available |
| Approved severity rubric and reviewed severity dataset | Not yet available |
| Downloaded source pool | 800 candidate versions: 609 Erbil and 191 Amedi collection records; suitability and building identity still require review |

The current CNN misses about 45.8% of the **reviewed crack pixels** in the development folds. That identifies recall as a weakness on those labels; it does not measure errors across entire walls. Unreviewed pixels were excluded, and some reviewed dark lines may still be joints, shadows or chipped edges. The three photographs have been repeatedly used to compare approaches, so they cannot now serve as an untouched test set.

Longer training was already tried: extending the replay experiment from four to twelve epochs, with 512-pixel evaluation windows, moved pooled Dice from about 68.56% to 69.23%. That modest improvement is recorded alongside unsuccessful texture and hybrid experiments. The final all-data checkpoint was trained separately; its fold models' scores are **not** an independent evaluation of that checkpoint. See [the complete experiment comparison](../REVIEWED_IMPROVEMENT_RESULTS.md).

### Define the target before measuring it

For the existing binary goal, use **foreground crack Dice**:

`Dice = 2 × TP / (2 × TP + FP + FN)`

An 80% Dice target means `Dice >= 0.80` on a specified evaluation set. It does not mean 80% IoU, pixel accuracy, four-class accuracy or severity accuracy. At Dice 0.80, the corresponding binary IoU is approximately 0.667. Predicting every currently reviewed pixel as intact already gives about 93.1% pixel accuracy, which is why that number must not replace crack Dice.

For a defensible future claim, define the target on **new, locked building groups with independently reviewed reference masks**, and report Erbil and Amedi separately as well as pooled. Agree the four-class and severity success criteria with the supervisor separately. No particular number of regions, epochs or downloaded photographs guarantees a score.

## 1. Agree consistent labels and obtain the missing rubric

The segmentation labels describe **what is visible at each pixel**. Severity labels describe an **image-level visual condition under an agreed rubric**. They are different tasks.

| Pixel ID | Label to review | Important boundary rule |
|---|---|---|
| 0 | Healthy/visibly intact masonry | A reviewed surface, not the default for anything unmarked |
| 1 | Visible crack | Distinguish fissures from mortar joints, carving, shadows and missing mortar; the legacy name `Structural Crack` does not prove structural significance |
| 2 | Surface erosion | Agree what counts as masonry material loss; rough texture, soil and missing render are not automatically erosion |
| 3 | Vegetation overgrowth | Confirm growth on/in masonry; a branch or plant merely in front of a wall is not sufficient |
| 255 | Ignore | Background, obstructions, unreviewed pixels and unresolved class boundaries |

Record decisions for plaster/render, mortar, rubble, overlaps and ambiguous boundaries before expanding the labels. Review the complete visible crack width and its surrounding surface, rather than selecting only the darkest center pixels. Keep masks at the source image dimensions, with an annotator, independent reviewer and version history. Preserve earlier accepted masks; corrections should be explicit reviewed revisions, not silent changes made to improve a score.

The user confirmed that the supervisor has not supplied a severity rubric. **Severity research training must wait for an actual approved, versioned rubric and reviewed grades.** The pending draft is a discussion aid; changing its status alone does not constitute approval. Do not derive low/moderate/high grades from predicted crack percentages. An unassessable image must not become a `none` grade. See [annotation rules](../ANNOTATION_PILOT.md) and [the severity workflow](../SEVERITY_WORKFLOW.md).

## 2. Add independent photographs and difficult examples from both sites

More regions in image 0750 add label detail but do not create another independent building. Extend coverage across verified buildings, materials, viewpoints, distances, lighting and weathering in **both Erbil and Amedi**. Replace unsuitable scenic, event or non-masonry images in the candidate pool. Preserve the source, author, license, source dimensions and hashes for every replacement.

Useful next review work is already prepared:

| Review step | Concrete action | What counts as progress |
|---|---|---|
| Review the nine pending source photographs | Inspect IDs **0018, 0021–0024, 0725, 0736, 0751 and 0777** at native resolution; record keep, exclude or uncertain, with a reason | Confirmed usable sources, not approved masks |
| Resolve related views | Compare 0018/0021, the Qubehan school views 0022–0024, and 0751 against existing 0750 | Verified building groups; these nine photographs are not nine independent buildings |
| Continue binary region review on new sources | Keep nine regions per session, alternating **5 crack/4 intact** and **4 crack/5 intact**, following the user's existing workflow | Confirmed cracks and intact examples from different photographs; do not invent cracks to fill the quota |
| Include difficult non-crack examples | Review mortar joints, shadows, carving, brick edges, textured stone and weathered surfaces | Useful negative examples **only when the reviewer can establish their class**; otherwise use ignore |
| Expand to all four classes | Review complete relevant masonry areas that contain real erosion and vegetation examples, with surrounding intact surfaces and cracks | Dense, consistently reviewed four-class masks; binary batch approval does not approve erosion or vegetation |

Start with [the two-site review guide](../TWO_SITE_DATA_REVIEW.md). Its pending worksheets deliberately have blank labels and unresolved building groups. Do not mark them approved just to make training run. The local review pack is referenced there; it requires the project's local source archives and may not be present in a source-code-only clone.

For a small reference subset, review **every eligible pixel within a defined masonry region**, not only selected easy patches. Include the difficult intact surroundings so false positives can be measured. Use a second reviewer to resolve disagreements. Keep independent test annotation blind to model predictions where practical; model-assisted proposals used for training still need correction and recorded provenance.

## 3. Reserve whole buildings for a locked test

Before extracting patches or choosing models:

1. Verify each photograph's city and building, and identify other views, crops and near duplicates. File hashes alone do not prove independence.
2. Assign entire building groups to train, validation or test. Keep all views and crops of one building together, including related views of the three existing development photographs.
3. Include both cities in every split and plan adequate coverage of all four classes and severity grades. The validators' minimum checks are not a sample-size calculation or proof of balanced coverage.
4. Record and version the split, source hashes and masks. Reserve test outcomes until model, preprocessing, threshold and rubric choices are frozen.

The current segmentation validator checks file hashes, review fields, mask IDs, building split conflicts and site consistency. It does **not** establish that a person actually reviewed a mask, automatically recognize every duplicate scene, or certify building identity. These require human review. The severity validator also checks recorded duplicate groups and decoded-image duplicates; different views still need correct grouping.

Use the existing three Erbil photographs as development data. Do not move their regions into a newly named test split and claim independence. If a final test result leads to more tuning, disclose that reuse and obtain a new untouched test before making another final generalization claim.

## 4. Train the real target tasks, then diagnose specific failures

Once actual reviewed masks and building assignments exist:

```sh
python project.py check-segmentation-data data/annotations/training-manifest.csv
python eda_four_class.py data/annotations/training-manifest.csv --output target-eda-v1.json
python project.py train-segmentation data/annotations/training-manifest.csv --output local-results/four-class-target-v1 --epochs 30 --warmup 5
```

Use a fresh EDA filename and fresh experiment directory. The manifest follows [the segmentation template](../training-manifest-template.csv). The entry point requires both cities in each split, reviewed masks and all four classes in training. Inspect **pixel support, source-image counts and building counts**, not only the number of patches. The default training command compares U-Net, DeepLabV3+ and FPN on the same split; its settings are engineering defaults, not established optimums. Test evaluation stays off unless explicitly requested after choices are frozen. See [the training guide](../FOUR_CLASS_RUN_GUIDE.md).

Once the severity rubric and labels have actually been approved:

```sh
python project.py check-severity-data data/severity/manifest.csv --rubric data/severity/rubric-approved-v1.json
python project.py train-severity data/severity/manifest.csv --rubric data/severity/rubric-approved-v1.json --output local-results/severity-target-v1 --epochs 20 --random-init
```

These commands must currently fail on the pending rubric or empty templates. The severity baseline starts from random initialization; it does not silently reuse the crack model. The existing implementation rejects custom encoder initialization until its provenance can be checked against evaluation groups. Use the separate, explicitly gated severity test command described in [SEVERITY_WORKFLOW.md](../SEVERITY_WORKFLOW.md) only after freezing choices.

For each development experiment, write the hypothesis first and change one substantive factor at a time. Inspect whether errors come from faint cracks, missing context, domain differences, incorrect labels or class imbalance. Possible future controlled changes include training resolution/context, reviewed rare-class sampling, appropriate augmentation and loss weighting. These are **experiments to validate**, not improvements already implemented or guaranteed to work. Any external training data need compatible labels, provenance and overlap checks against reserved buildings.

## 5. Report complete evidence and keep uncertainty visible

For four-class segmentation, report the confusion matrix, class support, per-class Dice/IoU and mean IoU, with separate Erbil/Amedi results. Show intact-wall false positives and missed cracks, erosion and vegetation. Do not report only the best photograph or omit a difficult building. For binary comparisons, keep the same eligible reference pixels and report precision and recall alongside Dice.

For severity, report the confusion matrix, per-grade support/recall/F1, macro F1 and ordinal error in class steps when the approved rubric defines an order. A grade absent from the reference set is missing evidence. The current implementation reports undefined metrics where appropriate; always show coverage rather than interpreting an average without its class supports.

Keep source/mask hashes, preprocessing, split membership, seeds, versions and failed runs with the results. Record model selection separately from final testing. When enough independent buildings exist, repeated grouped development splits and building-level uncertainty estimates are useful prospective additions; resampling millions of correlated pixels does not create millions of independent observations.

Prediction probabilities are currently **raw and uncalibrated**. Calibration, an abstention policy for uncertain or unsuitable images, and a measured human-review workflow are prospective deployment work. They require separate development/calibration evidence and an untouched test of the resulting policy. Report both error and the proportion of cases accepted; hiding uncertain cases cannot justify an improved general accuracy claim. Never automatically convert confidence, colored-pixel area or a model grade into structural safety advice.

## What must be true before claiming higher results

- The improvement is measured against reviewed references on a declared dataset and metric, not against the model's own proposals.
- Both cities, independent buildings and relevant class/grade coverage are represented in the evidence claimed.
- Test groups were not used to train, select epochs, select thresholds or decide which candidate to report.
- Four-class segmentation and severity have their own evaluations; the binary crack score is not substituted for either.
- The final report states the limitations and distinguishes the trained final artifact from the fold models used for development.

The immediate useful next action is to review the pending two-site photographs, confirm building identities, and agree the class and severity rules. See [the full project workflow](../ALL_FOUR_PARTS.md) and [the completion contract](../COMPLETION_PLAN.md). No additional region count or parameter sweep can replace those requirements.
