# AON human stress-related affect analyses

This repository contains the code for the main human analyses of stress-related affect during movie viewing. The 87 regions and their atlas sources are listed in [`roi_87_nodes.csv`](roi_87_nodes.csv).

## Analyses and scripts

**High- versus low-affect intersubject correlation.** `scripts/compute_d_isc.py` calculates leave-one-participant-out ISC in sliding windows, compares the highest and lowest within-run affect thirds, ranks all 87 regions and computes participant-level paired contrasts. It also performs within-run circular-shift tests.

**Comparison with other emotions.** `scripts/compute_f_emotions.py` compares the AON ISC effect for stress-related affect with the effects for anger, disgust, happiness, sadness and surprise. It uses within-participant differences, exact sign-flip tests and Benjamini–Hochberg correction.

**Dynamic connectivity classification and regression.** `scripts/compute_fc.py` calculates seed-to-region dynamic connectivity and evaluates all 87 seeds with nested leave-one-run-out cross-validation. It classifies high versus low stress-related affect with logistic regression and predicts the continuous ratings with ridge regression. Centering and scaling parameters are estimated from the training runs within each fold.

**Classification permutation test.** `scripts/compute_g_permutation.py` tests the AON classifier using within-run circular shifts. Each shift repeats label selection, hyperparameter tuning and model fitting.

`scripts/isc_common.py` provides shared ISC and region-label functions. `run_all.sh` runs the analyses in order.
