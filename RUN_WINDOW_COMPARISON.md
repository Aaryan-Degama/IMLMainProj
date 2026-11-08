# Execution

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

Notebook interpreter: `.venv/bin/python`.

## Main experiments

```bash
python run_notebooks.py
python run_notebooks.py --start 3
python run_notebooks.py --start 3 --models rf xgboost
python run_notebooks.py --start 3 --models mlp
python run_notebooks.py --start 3 --end 3 --models lr rf
```

Model options: `lr`, `rf`, `xgboost`, `mlp`. Default: all models, notebooks 01–04.
Notebook 04 uses saved baseline results from 03. Preparation notebooks recompute
their outputs. Executed copies are stored in `results/executed_notebooks/`.

## Sensor ablation

```bash
python run_sensor_ablation.py
python run_sensor_ablation.py --models rf --windows 2
python run_sensor_ablation.py --models lr xgboost --sensors back
python run_sensor_ablation.py --summary-only
```

Default: LR, both durations, back-only and thigh-only. Combined-sensor
results are read only. Ablation outputs: `results/sensor_ablation/`.
`--summary-only` loads completed runs for the selected configurations.

## LR/RF learning curves

```bash
python run_learning_curves.py
python run_learning_curves.py --models lr --windows 2
```

Defaults: both durations; 2, 4, 7, 10 and 13 training subjects; three nested
subject orders. The fixed validation subjects are unchanged and test subjects
are excluded. Metrics and PNG/PDF figures: `results/learning_curves/`.
Completed subset fits are cached separately from the main experiments.

## Caching

LOSO checkpoints include data, settings and package fingerprints.
`FORCE_RETRAIN=True` recomputes baseline fits. `FORCE_MLP=True` recomputes the
MLP experiment. `REBUILD_WINDOWS=True` rebuilds both window caches.
