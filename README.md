# HARTH activity recognition

Comparison of window duration and sensor placement for subject-independent
activity recognition using accelerometer features.

## Method

- Non-overlapping 1s/2s windows at 50 Hz; majority activity labels.
- 58 time-domain features; 24 cached frequency features are excluded.
- LR, RF and XGBoost with fixed settings and training-fold preprocessing.
- LOSO evaluation over 21 usable subjects: 20 train, 1 test per fold.
- S006 is excluded by the cadence check (10 ms observed, 20 ms expected).
- Primary score: mean subject-level macro F1 over 12 activities; absent classes score zero.
- LR sensor ablation: 58 combined, 29 back-only and 29 thigh-only features.
- MLP/XGBoost curves: separate 13/4/4 subject train/validation/test split.

Subject IDs, timestamps and purity are excluded from predictors. Curve
checkpoints use minimum validation loss. Fixed-split test scores and LOSO
subject means are reported separately. Model selection on these results is
exploratory, without an independent final test.

LR is the reference ablation model: highest mean LOSO accuracy and macro F1
close to XGBoost. Other supported model ablations are outside the reported analysis.

## Files

| File | Purpose |
| --- | --- |
| `notebooks/01_understand_data.ipynb` | Data audit and activity distribution |
| `notebooks/02_windows_and_features.ipynb` | Windowing and feature extraction |
| `notebooks/03_classical_baselines.ipynb` | LOSO model comparison |
| `notebooks/04_error_analysis.ipynb` | Error analysis and learning curves |
| `notebooks/05_sensor_ablation.ipynb` | Single-sensor comparison |
| `har_utils.py` | Preprocessing and feature extraction |
| `comparison_utils.py` | Evaluation and checkpoints |
| `run_notebooks.py` | Notebook execution |
| `run_sensor_ablation.py` | Sensor-ablation execution |
| `run_learning_curves.py` | LR/RF learning curves by training-subject count |

Results: `results/window_comparison/`, `results/sensor_ablation/`.
LR/RF data-size learning curves: `results/learning_curves/`.
Executed notebooks: `results/executed_notebooks/`.
Commands: [execution guide](RUN_WINDOW_COMPARISON.md).

References: [HARTH](https://archive.ics.uci.edu/dataset/779/harth) ·
[Authors’ implementation](https://github.com/ntnu-ai-lab/harth-ml-experiments).
