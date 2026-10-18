# Similar projects and proposed solution

Research checked 18 October 2026. Repository documentation reviewed; no new model executed or benchmark independently reproduced. No verified ready-to-run model covering all four Kurdish masonry classes was found in this search.

| Source | Relevance | Limitation |
|---|---|---|
| https://github.com/dimitrisdais/crack_detection_CNN_masonry | Masonry crack segmentation, U-Net/FPN and dataset subset | Older stack; cracks only; GPL-3.0 code. Audit data terms separately. |
| https://github.com/DavidHidde/CNN-masonry-crack-tasks | Refactored masonry pipeline | Author warns results can differ from original; not tested here. |
| https://github.com/TolgaBkm/CH_Crack_Detection | Cultural-heritage crack instance segmentation, YOLOv8x at 640px | Crack only; README states Zenodo weights but direct usable weight download not verified. CC BY-NC-SA 4.0 stated. |
| https://github.com/Dhananjay42/crackseg9k | Approximately 9160-image crack benchmark and training code | Different domains; inspect component licenses and duplicated sources before use. |
| https://github.com/DCorradetti/REDAI_Id_Pattern | Stone deterioration recognition research | Not pixel-mask training labels; reuse license not established. |
| https://github.com/qubvel-org/segmentation_models.pytorch | U-Net, DeepLabV3+, FPN in our existing PyTorch stack | Encoder pretraining does not supply trained masonry damage heads. |
| https://github.com/cvat-ai/cvat | Manual and AI-assisted annotation and mask export | SAM proposals require corrections; export background must map to ignore rather than healthy. |

## Proposed route

1. Screen existing photos and verify building/site identity from sources. Replace unsuitable candidates. Do not count 800 downloads as 800 valid annotated images.
2. Establish annotation rules with examples for healthy masonry, visible crack, surface erosion and vegetation actually on the masonry. Structural significance cannot be inferred from every visible line. Define foreground/background and uncertain pixels as ignore 255; confirm how the instructor wants ambiguous structural-crack cases handled.
3. Select 30–50 suitable source photos as an initial workflow pilot, covering classes and multiple buildings. This is a proposed engineering milestone, not a statistically sufficient final dataset or a waiver of the required 800 annotated images.
4. Obtain checked pixel annotations, assisted by specialist crack proposals or interactive segmentation where useful. Review all masks; never reuse the failed CLIPSeg drafts as truth. External binary crack background is not necessarily healthy masonry; do not directly relabel it class 0 in our four-class task.
5. Group train/validation/test splits by building and source photo before patch extraction. Keep overlapping crops and duplicates together. Hold test annotations out of training and tuning; audit external dataset overlap.
6. Start with U-Net/EfficientNet-B0, proposed 256px patches and small batches. Reuse existing model library. Train on an available GPU for full runs; a bounded local smoke run may be possible, but this has not been tested with real training. A Moondream loading failure does not prove every segmentation CNN needs a new computer.
7. Compare FPN and DeepLabV3+ with the same splits and evaluation. Record class Dice/IoU, mIoU, pixel accuracy, precision/recall, ignore handling and failure cases. Do not claim structural safety or physical crack width without suitable evidence/calibration.
8. Expand annotation toward the required dataset size using model proposals corrected by reviewers. Verify whether instructor counts source images versus patches; crops do not silently fulfill an 800-source-image requirement.

Recommendation: retain our existing PyTorch implementation; use masonry-specific research for methods and external crack data for controlled experiments. The central missing resource is checked target-domain annotation, not another image-description model. Keep all publication citations and source-specific licenses when using third-party code/data.
