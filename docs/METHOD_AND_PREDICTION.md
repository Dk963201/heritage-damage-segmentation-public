# Method, parameters and prediction equations

This document describes the code that exists in this repository. The current trained model is a **binary crack pilot**. Four-class segmentation and visual severity are separate software paths; they do not yet have approved target training data and evaluated target models. See [Project scope](PROJECT_SCOPE.md) for the four project parts and current evidence.

## Python stack and what a model learns

The project uses Python, PyTorch, Segmentation Models PyTorch (SMP), NumPy, Pillow and OpenCV. The implemented four-class architectures share an **EfficientNet-B0 encoder**:

| Architecture | Main role of its decoder | Implemented output |
|---|---|---|
| U-Net | Reconstructs pixel detail using encoder features and skip connections | Four logits at every pixel |
| DeepLabV3+ | Combines context at multiple spatial scales with a decoder | Four logits at every pixel |
| FPN | Combines feature maps at several resolutions | Four logits at every pixel |
| Separate severity CNN | Pools the final EfficientNet feature map and applies a linear classification head | One score per approved image-level grade |

The learned parameters are convolution weights, biases, normalization parameters and the output-head weights. They are numerical values optimized from examples. Hyperparameters such as learning rate, epochs, patch size and batch size control that optimization; they are not damage measurements.

Write the network as $z=f_\theta(x)$, where $x$ is the prepared image, $\theta$ is the learned parameter set, and $z$ contains raw scores called **logits**. Training changes $\theta$ to reduce a loss on reviewed labels. Prediction holds $\theta$ fixed and transforms logits into class probabilities and decisions. A high probability is not automatically calibrated confidence.

## Shared normalization

After the workflow-specific image preparation, each RGB channel is normalized as:

$$
x'_c=\frac{x_c/255-\mu_c}{\sigma_c},\qquad
\mu=(0.485,0.456,0.406),\quad
\sigma=(0.229,0.224,0.225).
$$

The code uses stored image orientation; it does not silently apply an EXIF rotation. Photographs, masks and any region of interest must remain aligned.

The workflows deliberately have different spatial preparation. Using another workflow's resize, padding, CLAHE setting or class order can change predictions even when the weights load successfully.

| Workflow | Spatial preparation | Contrast/padding behavior |
|---|---|---|
| Selected binary crack predictor | Native-resolution, non-overlapping 512-pixel grid by default | No CLAHE; repeat edge pixels at bottom/right boundaries |
| Four-class segmentation | Native-resolution patches; final grid starts are aligned to cover the far image boundary | Optional tile-level CLAHE from the training protocol; PIL black padding for images smaller than a patch; average overlapping logits |
| Severity classifier | Resize the entire image to fit a square while preserving aspect ratio | Bilinear resize, centered black padding, no CLAHE; default square is 224 pixels |

## Four-class segmentation: training

The source is [train_four_class.py](../train_four_class.py). Reviewed masks use IDs 0 healthy masonry, 1 visible crack, 2 surface erosion and 3 vegetation overgrowth. ID 255 is excluded from the loss and evaluation. The legacy name `Structural Crack` in training metadata is not a structural diagnosis.

### Current defaults

| Parameter | Default / behavior |
|---|---|
| Architectures | U-Net, DeepLabV3+, FPN |
| Encoder | EfficientNet-B0 |
| Initialization | ImageNet encoder weights by default; `--random-init` uses random weights and avoids that pretrained-weight download |
| Patch size | 256 × 256 native pixels |
| Batch size | 2 patches; incomplete final training batch is dropped |
| Epochs | 30 |
| Optimizer | Adam |
| Base learning rate | $10^{-4}$ |
| Warmup | 5 epochs, linear |
| Seed | 42, reset for each architecture |
| Training augmentation | Paired horizontal/vertical flips; brightness and color factors sampled from 0.85–1.15 |
| CLAHE | Enabled by default: LAB lightness channel, clip limit 2, 8 × 8 tile grid; `--no-clahe` disables it |
| Selection | Highest validation mean IoU; earliest epoch retained on ties |
| Test | Off by default; `--evaluate-test` explicitly opts in after choices are frozen |
| Runtime | CPU in the current four-class implementation |

The learning rate at one-based epoch $e$ is:

$$
\eta_e=\eta_0\min\left(1,\frac{e}{E_{\mathrm{warmup}}}\right).
$$

The network returns four logits $z_{i,c}$ for pixel $i$. Softmax gives:

$$
p_{i,c}=\frac{\exp(z_{i,c})}{\sum_{k=0}^{3}\exp(z_{i,k})}.
$$

Let $V$ be reviewed pixels with label other than 255, and $y_i$ be the true class. The implemented focal term uses $\gamma=2$, with no additional class-specific alpha weight:

$$
L_{\mathrm{focal}}=-\frac{1}{|V|}\sum_{i\in V}(1-p_{i,y_i})^2\log(p_{i,y_i}).
$$

For each class present in the current target batch, the soft Dice term is:

$$
D_c=\frac{2\sum_{i\in V}p_{i,c}\mathbf{1}[y_i=c]+\epsilon}
{\sum_{i\in V}p_{i,c}+\sum_{i\in V}\mathbf{1}[y_i=c]+\epsilon},
\qquad \epsilon=10^{-6}.
$$

The training loss is $L=L_{\mathrm{focal}}+\operatorname{mean}_{c\ \mathrm{present}}(1-D_c)$. Ignored pixels contribute to neither numerator nor denominator. This differentiable training loss is distinct from the hard-mask Dice reported during evaluation.

The [manifest validator](../validate_training_manifest.py) checks reviewed masks, source identity and split/group constraints. The [project entry point](../project.py) additionally requires both cities in every split. Data validation does not manufacture labels or prove that a reviewer correctly identified a crack.

## Four-class segmentation: prediction

Use [predict_four_class.py](../predict_four_class.py) only with a compatible **four-class** checkpoint and its matching training protocol. Binary weights cannot be substituted. New training runs produce a sidecar binding the checkpoint hash to the protocol, architecture and class count. Preserve that file; a legacy checkpoint without it has an unverified preprocessing association.

```mermaid
flowchart LR
    A[RGB photo and optional masonry ROI] --> B[Read matching checkpoint and protocol]
    B --> C[Native image tiles]
    C --> D[Protocol CLAHE and normalization]
    D --> E[Four-class CNN logits]
    E --> F[Average overlapping logits at each pixel]
    F --> G[Softmax and argmax]
    G --> H[Mask, aligned overlay, probabilities and metadata]
```

For dimension $n$ and patch size $P$, the trainer's starts are the unique sorted values of the regular $P$-spaced grid up to $n-P$, plus $\max(n-P,0)$. Thus the final tile may overlap the preceding tile. Small images are black-padded by PIL to a full patch.

If pixel $i$ occurs in $m_i$ tiles, average its logits first:

$$
\bar z_{i,c}=\frac{1}{m_i}\sum_{t:\,i\in t}z^{(t)}_{i,c},
\qquad
\hat y_i=\arg\max_c\bar z_{i,c}.
$$

Softmax is applied to $\bar z$ to export probabilities. This is **not** an average of per-tile softmax probabilities. Inference does not resize the source, search thresholds, clean up masks or create new reviewed labels.

An optional binary masonry ROI must match the source dimensions. Outside it, the exported class mask is 255, probabilities are NaN and the overlay retains the original pixels. Inside the ROI, a descriptive class fraction is:

$$
f_c=\frac{\sum_{i\in R}\mathbf{1}[\hat y_i=c]}{|R|},
$$

where $R$ is the supplied included region. Without an ROI, $R$ is the entire image, including background. These are predicted **pixel fractions**, not calibrated physical areas, severity grades or structural safety measures. See [the full inference contract](../FOUR_CLASS_INFERENCE.md) for output files and limits.

## Current binary crack pilot

The selected experimental model is U-Net/EfficientNet-B0 with one output channel. Its sigmoid decision is:

$$
p_i=\sigma(z_i)=\frac{1}{1+e^{-z_i}},\qquad
\hat y_i=\mathbf{1}[p_i\geq0.5].
$$

The [selected replay experiment](../train_reviewed_replay_long.py) starts each fold from the recorded public-masonry checkpoint. It freezes the encoder and batch-normalization running statistics, then adapts the remaining weights using a mixture of reviewed target patches and public training patches.

| Selected pilot setting | Value |
|---|---|
| Target photographs | 0650, 0450, 0750; all Erbil Citadel |
| Evaluation split | Three whole-photo holdout folds; evaluation image excluded from its fold's target training |
| Training patch / evaluation tile | 128 / 512 native pixels |
| Batch | 4 patches: 2 sparse target, 2 public |
| Training | 12 epochs × 32 optimizer steps per epoch |
| Optimizer | AdamW, learning rate $2\times10^{-5}$, weight decay $10^{-3}$ |
| Public replay pool | 640 fixed crops from 80 UK training photographs; 20 public validation photographs excluded from this pool |
| Augmentation | Paired rotations/flips |
| Seed / decision threshold | 333 / 0.5 |

For binary labels $y_i\in\{0,1\}$ and valid pixels $V$:

$$
L_{\mathrm{BCE}}=-\frac{1}{|V|}\sum_{i\in V}\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right],
\qquad
L_{\mathrm{Dice}}=1-\frac{2\sum_{i\in V}p_i y_i+1}{\sum_{i\in V}p_i+\sum_{i\in V}y_i+1}.
$$

The replay loss is:

$$
L=L_{\mathrm{BCE,target}}+0.25L_{\mathrm{Dice,target}}
 +L_{\mathrm{BCE,public}}+L_{\mathrm{Dice,public}}.
$$

Unreviewed target pixels remain ignored. Regions 334–336 are not training/evaluation labels. Public replay supplies additional reviewed non-crack/crack context; it does not add target-site severity or four-class labels.

[predict_reviewed_cnn.py](../predict_reviewed_cnn.py) uses a different grid from the four-class tool: starts are 0, 512, 1024, … with **no overlap**, and short bottom/right tiles repeat edge pixels. It exports the full source shape. Its output PNG uses 255 for predicted crack, while the corresponding binary **training** mask uses 255 for ignored pixels. Keep prediction artifacts separate from annotations.

The final all-three-image checkpoint was trained after the fold comparison. The fold scores belong to separate held-out fold models; they do not independently evaluate this final checkpoint. Saved weights and data must be available separately from source code; consult the [repository guide](../README.md) and [improvement artifact list](../REVIEWED_IMPROVEMENT_RESULTS.md).

## Severity classification is a separate model

[severity_pipeline.py](../severity_pipeline.py) predicts one image-level grade. It does not turn a crack percentage into low/moderate/high, and it does not reuse the binary mask as a severity label.

Let the final encoder feature map be $F\in\mathbb{R}^{d\times h\times w}$. Global average pooling and the linear head give:

$$
g_j=\frac{1}{hw}\sum_{u=1}^{h}\sum_{v=1}^{w}F_{j,u,v},
\qquad z=Wg+b,
\qquad p_c=\frac{e^{z_c}}{\sum_k e^{z_k}},
\qquad \hat y=\arg\max_c p_c.
$$

Training uses ordinary multiclass cross entropy:

$$
L_{\mathrm{severity}}=-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}.
$$

The approved rubric defines the class order. The pending draft proposes `none → low → moderate → high`, but these are not approved research labels yet. The current loss does not explicitly optimize ordinal distance; ordinal error is reported during evaluation.

| Severity setting | Current default / rule |
|---|---|
| Architecture | EfficientNet-B0 encoder, global average pooling, linear head |
| Image input | Full RGB photo fitted into 224 × 224 with bilinear resize and centered black padding |
| Initialization | Random weights throughout; custom pretrained initialization disabled until its source provenance can be checked for evaluation leakage |
| Optimizer | AdamW, learning rate $10^{-4}$, weight decay $10^{-4}$ |
| Epochs / batch / seed | 20 / 4 / 333 |
| Augmentation / runtime | None / CPU with deterministic PyTorch algorithms |
| Selection | Highest validation macro F1; earliest epoch on ties |
| Test access | Explicit opt-in; training and original model-selection validation groups excluded from test |

Training requires a released, approved rubric with a named approver, reviewed image-level labels, verified hashes and separate building/duplicate groups, with Erbil and Amedi in each split. A missing model returns an unavailable error. No real severity model, accuracy or grades are supplied by the current project. Read [the severity workflow](../SEVERITY_WORKFLOW.md).

## How performance is evaluated

For class $c$, use one-vs-rest counts from a confusion matrix whose rows are true classes and columns are predicted classes:

$$
\mathrm{Precision}_c=\frac{TP_c}{TP_c+FP_c},\qquad
\mathrm{Recall}_c=\frac{TP_c}{TP_c+FN_c},
$$

$$
\mathrm{Dice}_c=\mathrm{F1}_c=\frac{2TP_c}{2TP_c+FP_c+FN_c},\qquad
\mathrm{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c}.
$$

Segmentation mean IoU is the mean over classes with defined IoU. Severity macro F1 is the mean over classes with defined F1, with class support, absent true classes and contributing-class count reported. A class absent from both truth and predictions has undefined Dice/F1/IoU, represented as `null`; it is not credited as perfect. If it has false positives or false negatives, the corresponding denominator is nonzero and the score can correctly be zero. Precision and recall have their own denominator checks.

For $N$ severity images and ordered class indices:

$$
\mathrm{Ordinal\ MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat y_i-y_i|.
$$

This measures class-step error under the rubric, not physical damage. Empty site slices have undefined aggregate metrics. The code also reports class support and confusion counts so a high aggregate cannot hide missing coverage.

For the binary pilot, pooled Dice is computed after summing TP, FP and FN across all reviewed held-out pixels. It is not the arithmetic mean of each photograph's Dice. The current best reused-development counts are:

| TN | FP | FN | TP | Reviewed pixels |
|---:|---:|---:|---:|---:|
| 164,995 | 280 | 5,601 | 6,617 | 177,493 |

These yield **69.23% crack Dice, 52.94% IoU, 95.94% precision and 54.16% recall**. Many reviewed crack pixels are still missed. Pixel accuracy is about 96.69%, but predicting intact everywhere already reaches about 93.1%; accuracy alone is therefore a misleading success target here. **80% Dice remains unmet.** See [all improvement experiments](../REVIEWED_IMPROVEMENT_RESULTS.md) for retained unsuccessful candidates and limitations.

## What makes the result credible

Use reviewed reference labels, not predictions, to measure performance. Verify sources and grouping before splitting; photographs/crops/near duplicates of one building must stay together. Train on training data, select settings on validation, and keep the final test reserved until choices are frozen. Report per-class and per-site outcomes, including weak cases and undefined metrics.

The current three Erbil photos have been reused for development and contain selective, darkness-assisted labels rather than fully reviewed surfaces. Their scores cannot establish performance on new buildings, Amedi, erosion, vegetation or severity. SHA-256 records establish file identity, not annotation correctness. Passing synthetic software tests establishes that code runs and obeys contracts; it is not a scientific accuracy result.

Runnable commands and the remaining data gates are in [ALL_FOUR_PARTS.md](../ALL_FOUR_PARTS.md), [FOUR_CLASS_RUN_GUIDE.md](../FOUR_CLASS_RUN_GUIDE.md), [FOUR_CLASS_INFERENCE.md](../FOUR_CLASS_INFERENCE.md) and [SEVERITY_WORKFLOW.md](../SEVERITY_WORKFLOW.md).
