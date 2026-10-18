# Heritage study readiness

Checked: 2026-10-18T19:27:29.860140+00:00

**Research status: incomplete.** This report inventories local evidence; it does not run evaluation.

| Component | Software present | Research evidence still needed |
|---|---|---|
| Four-class segmentation | True | Reviewed four-class masks, trained target models and held-out results |
| Severity classification | True | Approved rubric, reviewed severity labels and held-out results |
| Both cities | True | Verified buildings and reviewed labels from both cities |
| Three CNN architectures | U-Net, DeepLabV3+, FPN training code | Fair target-domain comparison after annotations |

Candidate records: 800; collection-site counts: {'Amedi': 191, 'Erbil': 609}.

Existing binary development crack Dice: **69.23%**. This is not four-class or severity accuracy.

Severity rubric: **pending**.

## What must happen next

- Reviewed four-class masks and verified building groups from both Erbil and Amedi.
- An approved visual-severity rubric and independently reviewed image-level severity labels.
- Target-domain training and a frozen, independent test for segmentation and severity, reported by site.
- A final research report based on those experiments and confirmation of the lecturer's dataset requirements.

See ALL_FOUR_PARTS.md for commands and the current review packet.
