# Ten-photo visual screening completed

Reviewed individually in this conversation on 18 October 2026 using displayed image previews, up to about 2048 pixels on the long side. This is Automated vision-assisted visual screening, not a local Moondream run, an independent human review, or a measured accuracy benchmark. Full-resolution crops and expert annotation remain pending. No photos were deleted, cropped, or given pixel masks.

**Result:** 1 close-up candidate, 6 crop candidates, 2 excluded from detailed labeling, 1 scope-review hold. This purposive sample cannot estimate suitability across all 800 photos. The earlier provisional reference is preserved; this follow-up adds action decisions, not an independent agreement score.

| ID | Decision | Observation | Next action | Caution |
|---|---|---|---|---|
| 0001 | scope_review | Carved natural rock relief with pitting and small plants. | Hold until supervisor confirms carved rock is in scope. | Natural pores and carved contours must not be automatically called erosion or cracks. |
| 0010 | exclude_from_detail_labeling | Distant partly ruined building with antennas on a rocky hill. | Keep as context only; obtain closer masonry photos. | Wall detail is insufficient for reliable fine crack masks; site identity is not established visually. |
| 0030 | exclude_from_detail_labeling | Mountain and valley landscape; no usable masonry visible. | Exclude from masonry training selection; preserve original. | Landscape vegetation and rock are not masonry damage. |
| 0192 | crop_candidate | Brick courtyard walls, arches and steps; rear wall has visible irregularities. | Inspect crops of left brick wall and rear wall; exclude sky, people, seating and vegetation. | Ordinary joints, shadows and cables can resemble cracks; crops require resolution review. |
| 0250 | crop_candidate | Brick street walls and arch; a jagged vertical opening is visible on the left middle-distance wall. | Review a crop around the vertical opening and separate foreground wall crops. | Opening cause and structural significance are unconfirmed; tree above wall is not necessarily wall overgrowth. |
| 0350 | crop_candidate | Brick building above an earth slope; crack-like separations visible near upper right brick corner. | Crop building wall near the brick/plaster junction; exclude slope and wires. | Soil erosion in foreground must not become masonry erosion labels; structural severity unconfirmed. |
| 0450 | crop_candidate | Stone arches and columns, brick walls, plaster, doors and windows. | Separate stone-column/arch crops from brick-wall crops; exclude openings and wood. | Construction joints, stone texture and discoloration are not automatically cracks or erosion. |
| 0550 | crop_candidate | Brick facade with patchy plaster, openings and balcony; horizontal separation visible in lower central plaster. | Crop central plastered wall and adjacent brickwork; mask fence, sky, plants and openings as appropriate. | Surface loss and separation are candidates for inspection, not verified structural damage; foreground plants are not proven wall growth. |
| 0650 | close_up_candidate | Uneven brickwork, missing render, open gaps and crack-like lines; dry foreground plants. | Prioritize for manual pixel annotation after checking full-resolution image. | Differentiate missing mortar, joints and cracks; foreground plants do not establish vegetation growing in masonry. |
| 0750 | crop_candidate | Brick facade over rough stone base; damaged-looking roof-edge render and visible surface lines. | Crop roof-edge detail and separate stone/brick wall regions. | Rough dressed stone texture is not automatically erosion; joints and repair seams need review. |

## How to use this review

Start manual annotation with 0650. Evaluate proposed crops for the six overview images at original resolution before accepting them. Keep 0001 on hold for scope approval, and omit 0010/0030 from the detailed damage-labeling selection. Maintain source-photo/building groups when splitting train and test sets so crops from one scene cannot leak across splits.

The four project damage classes remain unverified. No structural safety, severity, or cause can be established from this screen. Do not label all unmarked masonry healthy. See `visual-screening.csv` for source links, original download hashes and reviewed derivative hashes. Existing failed CLIPSeg draft masks remain unverified and must not be treated as ground truth.
