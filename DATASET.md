# Dataset documentation

**Source:** PlantVillage Dataset (Hughes & Salathe, 2015; Mohanty et al., 2016), GitHub mirror `spMohanty/PlantVillage-Dataset`, folder `raw/color`, tomato classes only.
**Use here:** research/educational prototype. Check the dataset's license/terms before commercial use.

**Labels discovered (verified from folder names, not assumed):** bacterial spot, early blight, healthy, late blight, leaf mold, mosaic virus, septoria leaf spot, spider mites, target spot, yellow leaf curl virus (10 classes; the two virus classes were included because they exist in the dataset). Class counts and the integrity report are generated into `data/processed/validation_report.json` by `scripts/prepare_dataset.py`.

**Integrity checks:** every file must decode, be >= 64 px on both sides, and not be a byte-identical duplicate (duplicates are dropped; cross-class duplicates are counted).

**Splits:** stratified 70/15/15 (seed 42), saved in `data/processed/manifest.csv`. Training additionally caps images per class for CPU speed; validation is capped for calibration speed; the **test split is complete and untouched** until the final evaluation.

**Out-of-scope check set:** healthy leaves of other crops from the same repository (apple, maize, grape, bell pepper, potato, strawberry) in `data/raw/ood_leaves/`, used only to *measure* abstention - never for training or threshold selection.

**Known issues:** uniform lab backgrounds; class imbalance (yellow leaf curl virus is the largest class); likely near-duplicate leaves across splits; no field images. Field validation is required.
