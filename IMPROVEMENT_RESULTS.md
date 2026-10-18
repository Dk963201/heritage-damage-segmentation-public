# Attempt to reach 80% crack Dice

**Result: final Dice68.45%, up15.63 percentage points from52.82%. The80% target was not reached.**

18 October 2026. Target: foreground pooled crack Dice >=80% on the unchanged 20 UK validation center crops at probability threshold0.5. These are development results; no independent test or Kurdish four-class accuracy is claimed. Building/scene grouping is unavailable and repeated evaluation makes this a tuning set.

## Completed first improvement run

20 additional training epochs, starting from the saved 5-epoch model. Final Dice **62.17%**, IoU **45.11%**, versus prior Dice52.82%, IoU35.89%. False positive pixels decreased from13,084 to5,160; true positive pixels rose from17,805 to18,806. The model still misses17,722 of36,528 labeled crack pixels. Best observed epoch19 was62.42%; the published checkpoint/result is the final epoch20. **80% not reached by this run.**

A separate inference pass reran both checkpoints and reproduced the recorded pooled confusion matrices exactly. Split/crop equality was asserted. Visual comparisons show examples still missed completely (e.g. image091) and substantial remaining false negatives. This is not solved by the high overall pixel accuracy, because97.21% of pixels are background.

## Additional-crop experiment

Planned after the first run: five further epochs from its final checkpoint, same optimizer/loss/scheduler approach, one extra crack-centered native256px crop from each training source only. Validation remains exactly unchanged. No external images are ingested. Additional crops and their source coordinates are recorded separately.

## Sources and reproducibility

See [IMPROVEMENT_RESOURCES.md](IMPROVEMENT_RESOURCES.md) for primary resource links, limitations and first-run command. These resources informed the attempt; their pretrained models were not executed in this experiment.

Data and derived visualizations: Katherine A. Storey and Andrew L. Snowball, Annotated Image Dataset of Masonry Building Cracks, https://zenodo.org/records/18458458, CC BY4.0. Comparison images are cropped and overlaid adaptations. White masks are the published annotations; red overlays are model predictions.

Do not use proposed masks as verified Kurdish labels. The800-photo four-class dataset still requires corrected masks, source grouping and an untouched target test set.

Additional-source access note: the collection's OneDrive masonry-folder link could not be retrieved by the web reader during this run. Its81-photo claim is a source description, not a downloaded/verified dataset count. OmniCrack30k requests educational access for its consolidated data; no email was sent.

Second-run command:
```sh
.venv-ml/bin/python train_expanded_pilot.py --pretrained --balanced --unet-only --epochs 5 --output local-results/expanded-pilot
.venv-ml/bin/python evaluate_improved_pilot.py --root local-results/expanded-pilot
.venv-ml/bin/python predict_pilot_sample.py --root local-results/expanded-pilot
```

## Completed additional-crop result

Five additional epochs with160 training crops from80 sources finished at **68.45% Dice,52.04% IoU**. Final confusion matrix (rows truth, columns prediction): [[1266767,7425],[13656,22872]]. Precision75.49%, recall62.61%; 80% Dice not achieved. Threshold remained0.5. This is the final checkpoint, not a cherry-picked earlier epoch.

| Experiment | Final Dice | Final IoU | Training coverage |
|---|---:|---:|---|
| Prior baseline |52.82%|35.89%|80 center crops|
| Lower-rate + augmentation,20 more epochs |62.17%|45.11%|same80 center crops|
| Extra labeled regions,5 more epochs |68.45%|52.04%|160 crops from same80 sources|

## Remaining work toward80%

Expand diverse, licensed human-labeled masonry training data and audit duplicates/scene overlap; consider a crack-pretrained nnU-Net baseline after auditing its training corpus. Hold out an independent target-domain test set. Correct the Kurdish four-class masks before claiming target-task accuracy. Do not choose easier validation examples, count background accuracy as crack accuracy, or relabel predictions as ground truth. More training is a candidate experiment, not a guarantee of80%.

## Final checkpoint verification and Kurdish illustration

A separate inference pass reproduced final Dice68.45342312% and IoU52.03740359%, and confirmed exact validation split/crop equality. Per-image results are in `improvement-run2-comparison.json`. Full histories are preserved for both runs.

The final model was applied to the same Kurdish0650 preview. Visual inspection still shows proposals on open masonry joints and some foreground vegetation, while upper-left plaster crack-like lines remain unmarked. No verified target mask exists, so neither improved Kurdish accuracy nor an80% target-domain score can be inferred. See `improvement-0650-overlay.jpg`; red is unreviewed proposed crack, all other mask pixels are unknown. Source attribution for0650 is in `annotation-queue-sources.csv`.

Final checkpoint remains local at `local-results/expanded-pilot/unet-diagnostic.pt`; it is not included in this upload. All scripts, result JSONs, source links and qualitative examples are published.
