# Two-site dataset readiness and next source-photo review

The local photo inventory is available and hash-verified. It is **not yet an approved four-class or severity dataset**. The next review pack contains five Amedi and four Erbil photos, with blank annotation records and no model-generated labels.

Open [the source-photo review page](local-results/two-site-review-20261018/review.html) or [the nine-photo contact sheet](local-results/two-site-review-20261018/contact-sheet.jpg).

## Actual inventory audit

`prepare_two_site_review.py` read every image member from the existing ten image-version archives. All **800** members matched their recorded SHA256, byte count and image dimensions. Image headers were opened and verified; this is a file-integrity check, not a scientific review of every image or annotation.

| Evidence | Result |
|---|---:|
| Candidate image versions | 800 |
| Erbil source-collection label | 609 |
| Amedi source-collection label | 191 |
| Unique image IDs / downloaded SHA256 values / recorded original SHA1 values | 800 / 800 / 800 |
| Missing or mismatching archived files | 0 |
| User-reviewed binary source photos | 3, all Erbil |
| Reviewed binary pixels | 177,493 |
| Binary intact / crack pixels | 165,275 / 12,218 |
| Approved four-class rows found in audited task manifests | 0 |
| Severity labels present in existing reviewed-photo records | 0 |

The site counts describe collection metadata. The Amedi collection includes regional landscapes, events and non-building images; **191 does not mean 191 verified heritage-building examples**. Unique hashes do not prove independent buildings, scenes or near-duplicate groups.

The existing masks for 0450, 0650 and 0750 remain sparse binary selections. Unreviewed pixels remain 255. They are not complete masks, expert-approved structural damage labels or four-class ground truth. Pending regions **334–336** remain ignored and unchanged. No previous annotation was overwritten.

Manifest inspection covered root CSV task manifests and `data/annotations` CSVs, excluding synthetic fixtures, public binary datasets and prediction outputs. The detailed scope, input hashes and inspected files are recorded in `data-audit.json`; this does not claim an unrestricted search of files outside this project. No approved severity rubric was supplied in the project completion plan.

## Next nine source photos

| Review number | ID | Source collection | Review focus |
|---|---|---|---|
| 1 | 0018 | Amedi | Carved gateway and steps; distinguish joints from fissures |
| 2 | 0021 | Amedi | Another gateway view; compare identity with 0018 |
| 3 | 0022 | Amedi | Qubehan school courtyard and arches |
| 4 | 0023 | Amedi | Qubehan school masonry wall and vegetation |
| 5 | 0024 | Amedi | Qubehan school walls, rubble and annotation scope |
| 6 | 0725 | Erbil | Broad brick wall; decorative openings and mortar |
| 7 | 0736 | Erbil | Decorated facade; eligible surfaces versus background |
| 8 | 0751 | Erbil | Facade shadows and branches; compare with reviewed 0750 |
| 9 | 0777 | Erbil | Wall buttresses, support poles, joints and shadows |

Selection was purposeful: source titles followed by visual inspection of original thumbnails, before generating any pack predictions. No predictions were generated. The previously reviewed three photos and prediction-demo photos 0340 and 0756 were excluded. Initial candidates 0001 and 0004 were left out because a rock relief and approach made masonry annotation scope less clear.

**These are nine photos, not nine independent buildings.** Photos 0018/0021 likely show the same gateway; 0022–0024 likely belong to the same Qubehan school structure. Photo 0751 needs comparison with the previously reviewed 0750 building. All building groups and splits remain **UNASSIGNED** until a reviewer confirms identities. Keep every view, crop and near duplicate of a building in one split. The small review pack alone is not enough to establish balanced classes, severity grades, or a defensible train/validation/test split at both sites.

## What the user and supervisor need to review

1. Decide which photos are relevant and usable at native resolution. Keeping a photo does **not** approve any pixel mask or class label.
2. Confirm source location, building identity and related views. Record real groups before splitting.
3. Agree the four-class annotation rubric. Existing software IDs are 0 Healthy Masonry, 1 Structural Crack, 2 Surface Erosion and 3 Vegetation Overgrowth, with 255 for ignore. Define how plaster/render, mortar, natural rock, rubble, shadows, background and ambiguous boundaries are handled. The name “Structural Crack” does not establish structural significance from a photo.
4. Draw masks at the original image dimensions, with uncertain/unreviewed pixels ignored. Record an annotator and an independent reviewer. Check class coverage from reviewed masks rather than source titles.
5. Obtain the supervisor's approved, versioned **visual-severity rubric** before assigning severity labels. Do not invent low/medium/high thresholds from predicted crack percentage or colored area. A photograph alone does not establish structural safety or calibrated physical damage.
6. Only after review, prepare actual training manifests with resolved image paths, mask hashes, approved review states and confirmed building groups. Reserve test groups until model choices are frozen; report both site slices.

`pending-four-class.csv`, `pending-severity.csv` and `review-records.json` deliberately contain blank labels, blank masks and pending approval states. They are review worksheets, **not ready-to-train manifests**. Relevance approval must not be copied into mask or severity approval automatically.

## Files and source preservation

- `data-audit.json`: inventory counts, archive checks, known label state and limitations.
- `image-audit.csv`: all 800 per-image file checks.
- `source-list.csv`: nine selected photos, author, source URL, license, hashes, archive/member references and existing local file references.
- `contact-sheet.jpg`, `thumbnails/`, `review.html`: original-image review views; no masks or prediction overlays.
- `pending-four-class.csv`, `pending-severity.csv`, `review-records.json`: blank pending review records.
- `pack-verification.json`: generated-pack checks and preservation evidence.

Full-resolution source photos were **not copied again**. The pack references existing archive members and verified extracted files where available; only small display thumbnails were generated. Amedi full-resolution images remain in the original archive. Use the recorded archive/member when an annotation tool needs an extracted copy, then preserve its hash and dimensions.

The source manifest records 774 CC BY-SA 4.0 images, 21 CC BY-SA 3.0, one CC BY-SA 2.0 and four public-domain entries. Four public-domain records have no license URL; this is recorded, not silently filled in. Selected photos retain individual authors, source-page links and license links in the review page and source list. The displayed derivatives only resize images and add layout; their attribution does not imply damage-label approval.

## Reproduce and verify

From the repository root, using the existing environment:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python prepare_two_site_review.py --output local-results/two-site-review-new
PYTHONDONTWRITEBYTECODE=1 .venv-ml/bin/python -m unittest discover -s tests -p 'test_two_site_review.py' -v
```

Use a fresh output directory. The script rejects an existing review pack to protect human edits. It does not download files, generate masks, call a model, assign severity, approve records or change old labels. Five meaningful checks cover corrupt source hashes, unsafe archive paths, site balance and reviewed-photo exclusion, preservation of existing worksheets, and refusal to include pending region pixels in accepted data.

Software and inventory readiness do not establish research completion. Human annotation, an approved severity rubric, confirmed building groups and independent evaluation remain required.
