# Dataset card and annotation plan

## Target dataset — candidate photos acquired; annotations pending

Required scope: at least 800 high-resolution images of Kurdish heritage masonry, with pixel-level labels for healthy masonry, visible crack, surface erosion and vegetation overgrowth. Erbil and Amedi are the named target locations. Downloaded candidates: **800** (609 Erbil,191 Amedi collection labels), each decoded and hash-verified. **Approved four-class target-image count:0.** Separately, three Erbil photographs have sparse user-reviewed binary crack/intact selections through region333; these are not complete four-class masks. Collection-site labels and building groups still require visual verification. These are API-advertised image versions, not uniformly original-resolution files. See IMAGE_DOWNLOADS.md and ALL_FOUR_PARTS.md.

Proposed acquisition: request authorised existing imagery or arrange permitted photographs at safe, publicly accessible locations. Do not enter restricted structures. Record usage permission, image provenance and capture grouping. Remove personal identifiers and location-sensitive metadata from shared copies. The plan does not assert permission has been obtained.

## Downloaded pilot source

Storey, Katherine A., and Snowball, Andrew L. (2026). *Annotated Image Dataset of Masonry Building Cracks for Training Machine Learning Models for Crack Segmentation*, v1. DOI: https://doi.org/10.5281/zenodo.18458458. License: CC BY 4.0, verified in the record API on 18 October 2026. License: https://creativecommons.org/licenses/by/4.0/

The authors describe 100 UK residential masonry photographs with binary crack masks. Non-crack background is not equivalent to verified healthy masonry. This dataset lacks the required Kurdish sites and erosion/vegetation labels. Use it to test the binary pipeline, not as a substitute for the target dataset or evidence of four-class performance. Any derived masks must disclose transformations and retain attribution.

Archives and expected source checksums:

| File | Source MD5 (integrity only) |
|---|---|
| masonry_crack_images.zip | 434dd20461bc9ea5332beca8750e538e |
| masonry_crack_masks.zip | 55e28405cd2d716f79da0b5bf1dcea7c |

The audit records actual counts, dimensions, mask values and SHA-256. Visual annotation quality and building-level independence still require manual review. The source record alone does not prove those properties.

## Target mask convention

| ID | Class | Annotation rule |
|---|---|---|
| 0 | Healthy masonry | Visible intact masonry; exclude unknown/background regions |
| 1 | Structural crack | Visible crack pixels; do not infer hidden structural severity |
| 2 | Surface erosion | Visible surface material loss; distinguish lighting and stains |
| 3 | Vegetation overgrowth | Visible vegetation occupying or covering masonry |
| 255 | Ignore | Non-masonry, occlusion or uncertain annotation |

Use a single-channel PNG mask at the image dimensions. Document overlap priority and boundary rules before annotation; use ignore for unresolved regions. Pilot masks use **255 for crack**, unlike the target convention. Never feed the pilot into the target pipeline without an explicit, documented conversion.

## Data dictionary

`manifest.csv` fields: image_id (unique string), image_path, mask_path, source_url, license_or_permission, site, structure_id, capture_session, width_px, height_px, image_sha256, annotation_version, reviewer_status, split. Capture personally identifying details in neither filenames nor repository metadata.

`structure_id` must identify the physical structure independently of filename. Proposed train/validation/test proportions: 70/15/15 by independent groups, adjusted to actual structure counts; this is a planning assumption. Assign groups before patching or augmentation. No train/test split is asserted for the pilot because building identities are not verified.

## Quality checks

Verify count and pairing, image/mask dimensions, allowed mask IDs, unreadable files, exact and near duplicates, provenance and label agreement. Review a sample across all sites and conditions with a second annotator; resolve disagreements and retain version history. Report per-class pixel frequencies and per-site sample counts before training.
