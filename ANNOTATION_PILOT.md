# Kurdish annotation pilot

37 source previews were prepared and visually triaged by the assistant from contact sheets: **29 candidates for original-resolution/crop review, 1 exposure hold (0580), 7 context-only images**. This is not 37 accepted annotated images. The initial queue is Erbil-focused and must later add suitable Amedi photos. No four-class masks are approved or created. All originals remain in their source ZIPs.

Open `data/processed/annotation-queue/contact-1.jpg` through `contact-4.jpg` to review. Source links and hashes are in `annotation-queue.csv`. Running `prepare_annotation_queue.py` regenerates previews and resets the CSV to pending; preserve a reviewed copy before rerunning.

## Pixel labeling rules

- 0: visible masonry judged healthy by reviewer; never a default for unannotated pixels.
- 1: visible crack meeting agreed annotation criteria. Structural significance must not be guessed from photographs; unresolved cases require supervisor review.
- 2: verified masonry surface erosion; not rough stone texture, foreground soil, or automatically all missing render. Agree mortar/render treatment before annotation.
- 3: vegetation actually growing on/in masonry. Plants in front of the wall are not sufficient evidence.
- 255: background, openings, foreground obstructions, unlabeled or ambiguous areas.

Use original-resolution images for annotation, not the preview JPEGs. Preserve exact source dimensions in PNG masks. Start with 0650, then promising wall crops such as 0200, 0260, 0420, 0620, 0680 and 0700. These are review priorities, not verified examples of each class.

CVAT can export segmentation masks; explicitly map its exported colors/background to the above integer labels. An exported default background must become ignore, never automatically healthy. Do not accept boxes filled with one class as accurate crack masks. AI-generated proposals require inspection and correction along their boundaries. Record annotator, independent reviewer, date and uncertain regions. Do not sign off on another person's behalf.

Before final train/validation/test assignment, identify buildings/views and duplicates. Group source photos and all their crops together; do not randomly scatter related images across splits. Count source images separately from patches. The project's 800-image requirement remains outstanding.

## Immediate blocker

There are no independently checked four-class target masks. A teacher or qualified reviewer must resolve class ambiguities and approve the masks. The assistant has prepared the queue and rules but cannot truthfully mark them human-verified. Training results from the separate UK binary pilot do not fulfill this requirement.
