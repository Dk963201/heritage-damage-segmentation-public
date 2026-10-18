# Project explained: purpose, mathematics and your role

## What we are building

A Python research system that takes a photograph of heritage masonry in Erbil or Amedi and predicts a class for each pixel. This is semantic segmentation. Compare U-Net, DeepLabV3+ and FPN under the same evaluation protocol.

Expected outputs: a colored prediction overlay, a class mask and visible class-area summaries for expert review. These are intended outputs, not a claim that a complete inspection application has already been delivered.

| Mask ID | Class | Meaning |
| --- | --- | --- |
| 0 | Healthy masonry | Visibly intact surface under the agreed annotation rules; not proof of structural soundness |
| 1 | Crack | Visible crack pixels; current code calls this Structural Crack, but photographs alone cannot confirm structural significance |
| 2 | Surface erosion | Visible surface material loss under the agreed rubric |
| 3 | Vegetation overgrowth | Vegetation growing on/over masonry under the agreed rubric |
| 255 | Ignore | Background, occlusion, uncertain or unlabeled regions; excluded from loss and metrics |

The network predicts four classes. Ignore is a ground-truth exclusion rule, not a fifth prediction class. Class-area estimates should be restricted to relevant masonry regions; sky or unrelated objects must not be counted as healthy wall. Confirm overlap and boundary rules before labeling.

## Real-world application

A conservation team could use overlays to locate surfaces needing closer inspection, organize survey records, and support maintenance review. Comparisons over time require comparable viewpoints, scale and lighting or image registration. A percentage of damaged-looking image pixels is not physical wall area without calibration and is not an engineering severity or structural-safety rating. No institutional deployment or partnership is claimed.

## Python tools and prediction

The existing implementation uses Python, PyTorch, segmentation-models-pytorch, NumPy, OpenCV and Pillow. See [train_four_class.py](train_four_class.py) and [metrics.py](metrics.py).

Let X be an image and theta the learned network weights. The network produces four scores (logits) at each pixel:

$$z = f_\theta(X).$$

For pixel i and class c, softmax turns scores into probabilities:

$$p_{i,c}=\frac{\exp(z_{i,c})}{\sum_{k=0}^{3}\exp(z_{i,k})},\qquad \hat y_i=\arg\max_c p_{i,c}.$$

For example, probabilities [0.10, 0.75, 0.10, 0.05] predict crack. These scores are model probabilities, not calibrated guarantees of correctness. Our full-image evaluator averages logits from overlapping patches, then takes argmax. Softmax and argmax of the same averaged logits give the same winning class.

## How the model learns

The human-reviewed mask supplies the correct class y_i. Let V contain only pixels whose target is not 255, and let p_t be the predicted probability of the correct class. Our focal loss uses gamma = 2 and no additional alpha class weighting:

$$L_{focal}=-\frac{1}{|V|}\sum_{i\in V}(1-p_{i,y_i})^2\log p_{i,y_i}.$$

This gives less weight to already-easy pixels. For each class c, let g_{i,c} be 1 when the human label is c and 0 otherwise. The soft Dice score used during training is:

$$D_c=\frac{2\sum_{i\in V}p_{i,c}g_{i,c}+\epsilon}{\sum_{i\in V}p_{i,c}+\sum_{i\in V}g_{i,c}+\epsilon},\qquad \epsilon=10^{-6}.$$

The current implementation averages Dice loss only over classes present in the batch's valid ground truth (set C_present):

$$L_{Dice}=\frac{1}{|C_{present}|}\sum_{c\in C_{present}}(1-D_c),\qquad L=L_{focal}+L_{Dice}.$$

Backpropagation calculates gradients of this loss with respect to theta; Adam uses those gradients and moving averages to update the weights. An all-ignore batch is skipped. Training loss is a learning objective, not the reported test score.

## How we evaluate

Compare hard predicted masks against reviewed ground truth on withheld images. Each valid pixel contributes to a confusion matrix: rows are true classes and columns are predicted classes. For each class, TP means correct predictions, FP means pixels incorrectly assigned to it, and FN means true pixels missed.

$$IoU_c=\frac{TP_c}{TP_c+FP_c+FN_c},\qquad Dice_c=\frac{2TP_c}{2TP_c+FP_c+FN_c}.$$

$$PixelAccuracy=\frac{\text{correct valid pixels}}{\text{all valid pixels}},\qquad mIoU=\frac{1}{|C_{supported}|}\sum_{c\in C_{supported}}IoU_c.$$

The implemented metrics pool counts across evaluated source images. A class with no truth pixels and no predicted pixels has an undefined score (null), not a perfect score, and is excluded from the mean. False-positive predictions for an absent class produce a zero score for that class. Report class support alongside scores and ensure the final test has adequate examples of all target classes.

Example: TP=80, FP=20, FN=20 gives IoU=66.7% and Dice=80%. Thus 80% Dice does not mean 80% IoU or 80% pixel accuracy. A majority-class prediction can have high pixel accuracy while missing cracks.

Additional useful diagnostics, not currently returned by metrics.py, are:

$$Precision_c=\frac{TP_c}{TP_c+FP_c},\qquad Recall_c=\frac{TP_c}{TP_c+FN_c}.$$

Zero denominators require an explicit reporting convention. Per-building/site errors and uncertainty estimates are also planned analysis, not existing results.

## Experimental settings and fair comparison

Learned weights are model parameters. The settings below are hyperparameters: engineering defaults, not teacher-mandated values or established optimal settings.

| Setting | Current four-class default |
| --- | --- |
| Architectures | U-Net, DeepLabV3+, FPN |
| Encoder | EfficientNet-B0 with ImageNet initialization |
| Input patch | 256 x 256 pixels |
| Batch size | 2 |
| Optimizer | Adam |
| Learning rate | 0.0001 after five-epoch linear warmup; then constant |
| Training budget | 30 epochs |
| Loss | Focal gamma 2 + soft Dice |
| Seed | 42 |
| Augmentation | Paired horizontal/vertical flips; image brightness/color changes |
| Preprocessing | CLAHE by default, ImageNet normalization |
| Current execution | CPU |
| Checkpoint selection | Best validation mIoU |

Group all photos, crops and duplicates of the same building together before training. Proposed split proportions are 70/15/15 by independent building groups, adjusted to the actual number and composition of buildings. Proportions alone do not ensure independence. Tune on validation only; freeze choices before the final test. Repeated training seeds or grouped folds and confidence intervals remain future work.

## What we must achieve

1. Confirm the lecturer's exact dataset, source-resolution, deadline and severity requirements.
2. Produce the required suitable source images and reviewed masks with provenance, class rules and building IDs. Our documented assignment scope is at least 800; confirm whether downloaded rendered versions are acceptable or originals/field photos are required.
3. Audit class balance, duplicate images, label quality and split independence.
4. Compare the three models fairly and report held-out per-class Dice/IoU and mIoU with errors and limitations.
5. Deliver reproducible code, a demonstration of overlays, research report and source acknowledgments. Severity prediction needs its own expert rubric and labeled outcomes if required.

Current evidence: 800 candidate downloads (609 Erbil, 191 Amedi collection labels), zero approved four-class target masks, and sparse user-reviewed binary selections on three Erbil photographs through region333 (177,493 reviewed pixels). The best current binary target-development result is69.23% crack Dice; the older73.25% score uses a different public UK development set and is not directly comparable. Neither is a Kurdish four-class score or independent final test. The exploratory80% goal is not an agreed assignment pass criterion and has not been reached. See ALL_FOUR_PARTS.md for the expanded software workflow and the remaining data/rubric requirements.

## How you can help, in order

1. **Confirm scope with the supervisor.** Ask whether public downloaded photos satisfy the 800-image requirement; what counts as an original image; whether visible crack labeling is sufficient; whether severity prediction is required; and confirm deadline, marking rubric and permitted AI assistance. This document does not send any message.
2. **Screen images with me.** For each photo choose keep for closer review, context-only, or uncertain. Look for clear wall surfaces with enough detail. Intact walls are useful too. Do not guess labels from thumbnails.
3. **Help identify buildings.** Supply verified structure/wall identities when known. “Erbil” alone does not identify a particular building. Leave unknown identities unresolved until evidence is available.
4. **Label a small pilot.** Begin with 20–30 suitable images as a practical pilot, not a sufficient final dataset. In CVAT draw accurate pixel masks for the agreed classes; use ignore for uncertainty. Review AI suggestions before accepting them.
5. **Arrange independent review.** A lecturer or knowledgeable reviewer should resolve ambiguous erosion/crack/vegetation cases and inspect a representative sample. Record who reviewed which mask and what changed.
6. **Understand and present the experiment.** Explain inputs, labels, models, loss, withheld test buildings and Dice versus IoU. We can rehearse this together.

I can prepare images, review candidates with you, build/check the code, run feasible experiments, calculate scores and document reproducible results. Your biggest contribution is verified local knowledge, careful annotation and obtaining the lecturer's actual requirements. Neither of us should mark unreviewed model-generated masks as human-approved ground truth.

Continue with [ALL_FOUR_PARTS.md](ALL_FOUR_PARTS.md) and the pending two-site review packet. Earlier binary selections are already saved; regions334–336 remain pending and excluded. New photographs and four-class/severity labels require separate review and must not overwrite those earlier selections.
