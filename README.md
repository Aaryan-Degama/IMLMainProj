# HARTH classical ML experiments

Our question is: **which classical ML models and sensor features recognize activities reliably for a person who was not in the training data?** We turn short sensor recordings into features, compare models, and inspect their errors. Later experiments should follow from those errors rather than simply add more models.

Status: pipeline prepared only. Training, preprocessing, notebook execution and tests have not been run. HARTH is already present in `harth_project/data/harth/`; notebooks read those local CSV files directly. There is no download or extraction step. Dependency installation was previously stopped and may still need completion. No model results are claimed.

**Normal workflow: install the environment once, then run notebooks 01 → 02 → 03 → 04 from top to bottom. No preparation or training script commands are required.** Each notebook defines its own one-notebook helpers and imports functions reused across notebooks. Opening a notebook does not execute it, but running all cells in notebooks 01–03 performs their auditing, preprocessing or training work.

Compute: a modern laptop CPU is sufficient; no GPU is required. Recommend 16 GB RAM (8 GB may work with sequential runs) and 3–4 GB free disk space for the environment, archive, extracted data and outputs. Full 22-fold LOSO repeatedly trains each model; allow substantial CPU time. Runtime has not been measured. The four-fold ablations reduce cost. A cloud CPU machine is optional; GPU compute is only worth reconsidering if a CNN is added.

Use this folder and its existing HARTH data for the project. No PUC-Rio data, saved weights or reported scores enter these experiments.

## Work distribution for four members

Replace Member 1–4 with team names. These are suggested responsibilities, not a record of completed contributions. Each person owns a part and explains it to the rest of the team.

| Member | Milestone responsibility | Main code to understand | Work after the milestone | Expected handoff |
| --- | --- | --- | --- | --- |
| **1: Data and preprocessing** | Dataset version, activity labels, class counts, participant coverage, timestamp checks and window boundaries | Notebook 01's local `audit_data()`; shared `read_subject()` and `build_windows()` in `har_utils.py` | Help compare window lengths and trace errors back to recordings | Explain one reading versus one window; provide audit/count tables and account for discarded readings |
| **2: Features and representations** | Explain and manually check the 58 time features and training-only scaling | Notebook 02; `window_features()` and `feature_columns()` in `har_utils.py` | Lead notebook 05: frequency, sensor removal, magnitude-only features and PCA; coordinate balancing with Member 3 | Feature definitions, one worked window example and later a matched comparison for each representation |
| **3: Models and tuning** | Explain Dummy, LR, linear SVM, RF and XGBoost; document fixed baseline settings | Notebook 03; `models()` and `evaluate()` in `har_utils.py` | Lead notebook 06: RBF SVM and inner subject-wise model/parameter search defined in notebook 06 | Explain why each model is included, its main settings and the difference between fitting and tuning |
| **4: Evaluation and errors** | Check subject-disjoint splitting; interpret accuracy, macro F1, participant variation and confusion matrices | Notebook 04; `evaluate()` and `summarize()` in `har_utils.py` | Review notebooks 05–06 for fair comparisons and selection leakage | Identify difficult classes/participants, explain metric conventions and trace selected mistakes to raw data |

**Shared responsibilities:** agree on settings before running; use the same dataset version and label mapping; review another member's work; record actual contributions as work is completed. Everyone should understand the full pipeline. Report and poster writing can be assigned later.

Suggested review pairs: Member 1 checks Member 2's feature examples; Member 2 checks Member 1's window boundaries; Member 3 checks Member 4's prediction/label alignment; Member 4 checks Member 3's splitting and preprocessing.

## What everyone should study

Study these in order. Aim to explain each answer in your own words, not just describe a function call.

| Topic | What to understand | Check your understanding |
| --- | --- | --- |
| **HARTH and accelerometers** | Back/thigh sensor locations, three axes per sensor, acceleration in g, 50 Hz sampling and label meanings | Why can gravity and sensor orientation help distinguish postures? |
| **Reading versus window** | One row is a reading; one training sample summarizes successive readings | Why does a 2-second window contain 100 readings? What shape does the model actually receive? |
| **Window boundaries** | Non-overlapping windows, incomplete tails, invalid readings and timestamp gaps | Why must a window not cross a participant boundary or recording gap? |
| **Window labels** | Majority voting and purity; transitions can contain several activities | For 60 walking readings and 40 standing readings, what are the label and purity? |
| **Time features** | Mean, standard deviation, extrema, median, range, RMS, magnitude and SMA | Which describe orientation or variation? Can you manually calculate one example? |
| **Scaling and pipelines** | Preprocessing learns parameters and must be fitted within training folds | Why would scaling all participants before splitting leak information? |
| **Classical models** | Linear boundaries, margins, trees, bagging and boosting | How do LR, SVM, RF and XGBoost differ? Why include a majority-class dummy? |
| **Generalization** | A new recording from a known person differs from a recording from a new person | Why can a random split be optimistic even without overlapping windows? |
| **Imbalance and metrics** | Accuracy, precision, recall, F1, macro averaging and class support | Can accuracy be high while a rare activity has zero recall? |
| **Error analysis** | Confused activities, participant variation, rare classes and mixed windows | What observed failure would justify a particular feature improvement? |

### Member-specific deeper study

- **Member 1:** pandas loading/grouping, missing versus nonfinite values, timestamp differences, array shapes and continuity checks. Follow a subject CSV through to its window start rows.
- **Member 2:** axis versus magnitude information, RMS versus SMA and standardization. Later: FFT frequencies in Hz, spectral centroid/entropy and PCA variance retention.
- **Member 3:** regularization (`C`), linear versus RBF SVM, RF depth/leaf size, boosting depth/learning rate, seeds and sklearn Pipelines. Later: GridSearchCV and refitting.
- **Member 4:** LOSO, GroupKFold, confusion-matrix normalization, fixed-label versus present-class macro F1, fold means versus pooled predictions and nested validation. Study why choosing the best experiment can itself overfit evaluation results.

## Understand the data-to-model calculation

```text
Subject CSV: timestamp + six axes + activity label
    -> check readings and recording continuity
    -> 100 readings per 2-second window
    -> 58 time features per training sample
    -> hold out one participant
    -> fit preprocessing and classifier on the remaining participants
    -> predict the held-out participant
    -> repeat for all 22 participants and inspect errors
```

The six axes are back x/y/z and thigh x/y/z. We also calculate a magnitude signal for each sensor, giving eight signals to summarize.

| Feature | Definition |
| --- | --- |
| Mean | Average signal value |
| Standard deviation | Variation around the mean; population convention (`ddof=0`) |
| Minimum / maximum | Smallest / largest value in the window |
| Median | Middle value |
| Range | Maximum minus minimum |
| RMS | `sqrt(mean(signal²))` |
| Vector magnitude | `sqrt(x² + y² + z²)` at each reading, then summarized |
| SMA | `mean(abs(x) + abs(y) + abs(z))` for each sensor |

Seven statistics on eight signals give `7 × 8 = 56` features. Two sensor-level SMA features bring the time-feature count to **58**. The later frequency representation adds three features on eight signals, giving **82** total. Magnitude, RMS and SMA are different quantities.

Subject IDs, timestamps and purity describe samples but never become predictors. Window labels are assigned after boundaries are fixed, so the target does not determine where windows start or stop.

## Suggested team learning workflow

1. Everyone reads this guide and the UCI dataset description, then opens notebook 01.
2. Members 1 and 2 explain one raw recording, one window and one manual feature calculation to the group.
3. Member 3 explains the baseline model choices and settings; Member 4 explains exactly which participants are held out.
4. When runs are authorized and completed, the team works through notebooks 01–04 together and records findings. Until then, treat expected behavior as a hypothesis.
5. Use the errors to choose which later experiments deserve attention. Preserve failed improvements as evidence too.

Before presenting, every member should be able to explain the project question, the split, why scaling is training-only, what macro F1 measures and one limitation. Each owner should also demonstrate their input, central calculation and output.

## Milestone: notebooks 1–4

From `/Users/adithnr/IMLProj`:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r harth_project/requirements.txt
```

Select this virtual environment as the notebook kernel in your editor. Use Python 3.11 or newer. Then run each notebook in order using **Run All** (or execute its cells from top to bottom):

1. **Member 1 — `notebooks/01_understand_data.ipynb`:** reads the existing local data, computes and saves the audits, then explores subjects, labels and signals.
2. **Member 2 — `notebooks/02_windows_and_features.ipynb`:** builds and saves the 2-second windows, features and metadata, then explains their calculations.
3. **Member 3 — `notebooks/03_classical_baselines.ipynb`:** trains each baseline in a separate cell, saves its predictions/metrics and compares results. The random-split diagnostic also trains RF.
4. **Member 4 — `notebooks/04_error_analysis.ipynb`:** reads notebook 03's RF predictions and evaluates classes, confusions, participants and mixed windows. No extra training.

The notebooks use separate kernels and exchange saved files, not in-memory variables. They are sequential, not independent from scratch. With a shared project folder, finish one notebook before the next member starts. On separate machines, share the existing `data/harth/` folder, notebook 02's feature/window/audit files, and notebook 03's result/configuration files; alternatively rerun the earlier notebooks locally.

**Reruns are explicit:** notebook 01 reads existing data and refreshes audits; notebook 02 rebuilds and replaces feature files; notebook 03 retrains any model whose training cell is executed. Run only its analysis cells to revisit saved results. If windowing or features change, rerun the baseline cells before analyzing their predictions. Keep data, feature and result files from the same run together.

**Stop here for the milestone.** The intended outcome is a dataset audit, a reproducible baseline comparison and preliminary errors once the pipeline has actually run. Prepared code alone does not establish model performance. Leave notebooks 05–06 for later.

Notebook 02 includes visual walkthroughs using the actual saved windows: alternating shaded 2-second blocks over raw signals, per-reading activity labels, majority-vote counts for a mixed window when available, and a signal annotated with its summary features. These are code cells to execute later; no figures have been generated yet.

Milestone preparation creates only 2-second windows. Baseline models use only the 58 time features. Frequency features are cached for later, but no feature ablations or tuning are included in milestone training.

Notebooks 01–03 perform the work directly; notebook 04 reads their outputs. Functions used by only one notebook are defined in that notebook with explanations. Reused calculations stay in `har_utils.py`. `run_experiments.py` is an optional baseline-training alternative after notebooks 01–02; it does not run notebooks or provide separate preparation/ablation stages. There are no Colab paths or downloaded model weights.

## After the milestone: notebooks 5–6

5. `notebooks/05_feature_improvements.ipynb`: frequency, sensor, magnitude, balancing, PCA and window-length experiments on four subject-disjoint folds. Gravity/movement separation and axis correlations are documented follow-up candidates, not implemented results.
6. `notebooks/06_tuning_and_final_evaluation.ipynb`: small LR/RBF SVM/RF search using inner subject folds and outer LOSO, plus a matched comparison with the baseline. The tuning function is defined directly in notebook 06.

Both later notebooks have training disabled by default (`RUN_IMPROVEMENTS = False`, `RUN_TUNING = False`). Read and execute their function-definition cells, then enable the corresponding flag only when ready. Defining a function does not start training. They show a message when results do not yet exist.

Only notebook 05's enabled feature-experiment function prepares 4-second windows. Tuning uses the fixed milestone time features; selecting the best feature representation from notebook 5 would require moving that choice inside the inner search to evaluate the full selection procedure. The same participants have already informed exploration, so this is not an untouched external test set.

Notebook 05 asks whether frequency information, a different sensor representation, class weighting or PCA helps relative to a matched control. Notebook 06 asks whether a small model/parameter search helps when selection is done inside the training participants. In notebook 06, three inner subject folds select settings, then the held-out outer participant is evaluated. Different outer folds may select different models.

Gravity/movement separation, axis correlations and 1-second windows remain possible extensions, not implemented results. A CNN is optional and is not required to make the classical comparison meaningful.

## Which functions live where?

| Location | Functions | Why |
| --- | --- | --- |
| Notebook 01 | `audit_data()` | The initial participant and activity audit belongs to this notebook only |
| Notebook 05 | `run_feature_experiments()` | The later experiment sequence is used only here |
| Notebook 06 | `run_nested_tuning()` | The nested search is used only here |
| `har_utils.py` | Reading subjects, feature extraction/windowing, feature selection, model definitions, evaluation, summaries and environment recording | These are reused across notebooks or by shared windowing code |

Notebook 02's windowing/features and notebook 03's model definitions stay shared because notebook 05 reuses them. Notebook 04 already contains its analysis directly. The local function cells make the relevant implementation visible without duplicating shared algorithms. Run definition cells before cells that call them. Notebook order and saved-file dependencies are unchanged.

## File map and outputs to study

| File or generated output | Purpose | Main owner |
| --- | --- | --- |
| `data/harth/S*.csv`, notebook 01's `audit_data()` | Existing subject recordings and their quality/label audits | Member 1 |
| `har_utils.py` | Shared data checks, windowing, features, models and evaluation | All, by assigned functions |
| `run_experiments.py` | Optional baseline-only command-line alternative after notebooks 01–02 | Members 1–3 |
| Notebook 06's local `run_nested_tuning()` | Optional inner search and outer LOSO | Members 3–4 |
| `test_pipeline.py` | Synthetic checks for window boundaries and feature definitions; not yet run | Members 1–2 |
| `results/subject_audit.csv`, `class_counts.csv` | Data quality, imbalance and participant coverage | Member 1 |
| `results/window_audit_2s.csv`, `windows_2s.csv` | Discarded readings, window locations, labels and purity | Members 1–2 |
| `results/features_2s.pkl` | Cached 82-feature table; milestone selects the 58 time columns | Member 2 |
| `results/*_config.json` | Model settings and feature columns | Members 2–3 |
| `results/*_folds.csv`, `summary.csv` | Fold metrics and baseline/ablation summary | Members 3–4 |
| `results/*_predictions.csv` | Predictions traceable to subject, start row and timestamp | Member 4 |
| `results/environment.json`, `requirements-lock.txt` | Package versions recorded during preparation | All |

These generated outputs do not exist until their corresponding stage completes. Nested tuning writes its own files and checkpoints after each outer fold; notebook 06 checks for all 22 subjects before comparing full LOSO results. It reads its results directly rather than automatically updating `summary.csv`.

## Data and provenance

[UCI HARTH](https://archive.ics.uci.edu/dataset/779/harth), DOI [10.24432/C5NC90](https://doi.org/10.24432/C5NC90), CC BY 4.0. Credit: Logacjov, Kongsvold, Bach, Bårdstu and Mork. We use the fixed 22-subject release, six acceleration axes in g, sampled at 50 Hz. The existing CSV files are read directly; no downloads, archive extraction or checksum generation is performed by the notebooks. Large raw files and feature caches are excluded by the root `.gitignore`.

The [authors' implementation](https://github.com/ntnu-ai-lab/harth-ml-experiments) now contains HARTH v2 with 31 subjects. Do not mix its scores or label corrections with this release. No source code was copied from that repository.

## What we test

- Main baseline comparison: Dummy, LR, **linear** SVM, RF, XGBoost; identical time features, 2-second non-overlapping windows, all 22 LOSO folds.
- Exploratory feature experiments: time versus time + frequency; back versus thigh versus both; magnitude-only representation; class-balanced RF; LR with versus without 95%-variance PCA; 2 versus 4 seconds. These use four subject-disjoint folds for a less expensive first pass.
- An optimistic random-window RF split demonstrates the effect of sharing participants and adjacent recordings. It is not a generalization estimate.

All 12 activity codes are retained. Label voting happens after window boundaries are fixed. Windows do not cross subjects, invalid readings or timestamp discontinuities; incomplete tails are discarded and counted. No target-dependent boundary detection or pure-window filtering is used. Standardization and PCA are fitted inside each training fold. Identity, demographics, timestamps and label purity are never predictors. Hyperparameters are declared before runs; no test-fold tuning is performed.

## Metrics and interpretation

`results/summary.csv` contains the mean and sample standard deviation of **fold** macro F1, mean accuracy, and runtime. Fixed-label macro F1 averages all 12 labels, assigning zero when a class is absent. `present_class_macro_f1` separately averages labels present in each held-out fold. Pooled confusion matrices and reports answer a different, window-weighted question. Always specify which metric you use.

Each experiment saves its parameters and feature names, fold metrics and every held-out prediction. Prediction files include subject, start row, timestamp, winning label and purity so an error can be traced to the original recording. `window_audit_*` records sample losses. `environment.json` and `requirements-lock.txt` record the actual run environment.

Four-fold ablations are exploratory, not nested model selection. Do not choose the best settings here and present the same evaluation scores as unbiased final performance. Repeat promising changes under nested subject-wise validation. A 4-second window changes sample count and the voting targets, so it is a sensitivity check, not a paired window test. Synthetic tests cover gap handling, invalid readings, mixed labels, constant features and frequency units.

## Research checks and reasonable next improvements

Suggested reading order: first the UCI column/label definitions, then the HARTH paper's preprocessing, experimental setup and discussion, then the authors' implementation. Finally, trace the corresponding local helper functions. Compare ideas and evaluation rules rather than copying reported scores across dataset versions.

The [HARTH study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659926/) uses one-second windows, majority voting, engineered features and subject-wise evaluation. Its representation separates gravity and movement and includes cross-axis relationships. Our smaller feature set is a readable starting point, not a reproduction of its 161-feature implementation. Its merged-activity headline score is not comparable with our unmerged 12-label metrics.

The study also identifies imbalance as a limitation. That motivates the balancing ablation. Sensor removal and magnitude-only features test which information transfers between people; frequency features test whether periodic motion helps. PCA checks whether reducing redundant features helps the linear baseline.

Next: evaluate an RBF SVM; compare 1-second windows; add gravity/movement separation and a small set of axis correlations; tune only using inner subject folds. Confirm any improvement on the outer subjects and inspect rare-class recall. The optional CNN is deferred until these classical comparisons are understood.
