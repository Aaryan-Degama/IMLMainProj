"""Sensor ablation using cached windows and completed two-sensor LOSO results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from har_utils import ROOT, SEED, LABELS, ACTIVITIES, load_windows, feature_columns
from comparison_utils import (comparison_models, validate_windows,
                              load_baseline, run_baseline)

OUT = ROOT / 'results' / 'sensor_ablation'
MODEL_NAMES = {'lr': 'LR', 'rf': 'RF', 'xgboost': 'XGBoost'}

def check_predictions(meta, scores, predictions):
    """Require one held-out prediction per cached window and matching labels."""
    ids = predictions.window_id.to_numpy()
    if len(ids) != len(meta) or set(ids) != set(range(len(meta))):
        raise ValueError('Results do not cover each window exactly once.')
    expected = meta.iloc[ids].reset_index(drop=True)
    actual = predictions.reset_index(drop=True)
    if not expected[['subject', 'start_row', 'label']].equals(
            actual[['subject', 'start_row', 'label']]):
        raise ValueError('Saved predictions do not match the cached windows.')
    subjects = sorted(meta.subject.unique())
    if len(scores) != len(subjects):
        raise ValueError('Incomplete LOSO results.')
    for fold, subject in enumerate(subjects):
        row = scores[scores.fold == fold]
        p = actual[actual.fold == fold]
        if len(row) != 1 or row.test_subjects.iloc[0] != subject or set(p.subject) != {subject}:
            raise ValueError('Fold assignments differ from the reference LOSO split.')

def save_plots(summary, differences, class_scores, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    sensors = [s for s in ['both', 'back', 'thigh'] if s in set(summary.sensor)]
    durations = sorted(summary.window_seconds.unique())
    for metric in ['macro_f1', 'accuracy']:
        fig, axes = plt.subplots(1, len(durations), figsize=(6 * len(durations), 4), squeeze=False)
        for ax, seconds in zip(axes[0], durations):
            summary[summary.window_seconds == seconds].pivot(
                index='model', columns='sensor', values=metric).reindex(columns=sensors).plot.bar(ax=ax)
            ax.set(title=f'{seconds}s windows', ylabel=f'Mean subject {metric}', ylim=(0, 1))
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Sensors')
        fig.tight_layout()
        fig.savefig(figures / f'sensor_{metric}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, max(4, len(differences) * .35)))
    for i, row in differences.iterrows():
        ax.plot([row.ci_low, row.ci_high], [i, i], linewidth=2)
        ax.scatter(row.mean_delta, i)
    ax.set_yticks(range(len(differences)), [
        f'{r.model} {r.window_seconds}s {r.sensor}' for r in differences.itertuples()])
    ax.axvline(0, color='black', linestyle='--')
    ax.set(xlabel='Macro F1: single sensor − both sensors',
           title='Paired subject differences with 95% bootstrap intervals')
    fig.tight_layout()
    fig.savefig(figures / 'sensor_paired_differences.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    for (model, seconds), group in class_scores.groupby(['model', 'window_seconds']):
        fig, ax = plt.subplots(figsize=(10, 7))
        group.pivot(index='activity', columns='sensor', values='recall').reindex(
            index=list(ACTIVITIES.values()), columns=sensors).plot.barh(ax=ax)
        ax.set(title=f'{model}, {seconds}s: pooled class recall', xlabel='Recall', xlim=(0, 1))
        fig.tight_layout()
        fig.savefig(figures / f'{model}_{seconds}s_class_recall.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

def run_ablation(model_names=('LR',), durations=(1, 2),
                 sensors=('back', 'thigh'), summary_only=False, output_dir=None):
    out = OUT if output_dir is None else Path(output_dir)
    model_names = tuple(dict.fromkeys(model_names))
    durations = tuple(dict.fromkeys(durations))
    sensors = tuple(dict.fromkeys(sensors))
    if not model_names or not set(model_names).issubset(MODEL_NAMES.values()):
        raise ValueError('Unsupported baseline model selection.')
    if not durations or not set(durations).issubset({1, 2}):
        raise ValueError('Unsupported window duration.')
    if not sensors or not set(sensors).issubset({'back', 'thigh'}):
        raise ValueError('Unsupported sensor selection.')
    estimators = comparison_models()
    datasets, references = {}, {}

    for seconds in durations:
        X, meta = load_windows(seconds)
        validate_windows(X, meta)
        both = feature_columns(X)
        back, thigh = feature_columns(X, 'back'), feature_columns(X, 'thigh')
        if len(both) != 58 or len(back) != 29 or len(thigh) != 29:
            raise ValueError('Expected 58 combined features and 29 per sensor.')
        if set(back) & set(thigh) or set(back) | set(thigh) != set(both):
            raise ValueError('Sensor features are not a disjoint partition.')
        datasets[seconds] = X, meta
        for name in model_names:
            scores, predictions = load_baseline(X[both], meta, estimators[name], name, seconds)
            check_predictions(meta, scores, predictions)
            references[name, seconds] = scores, predictions
    out.mkdir(parents=True, exist_ok=True)
    all_scores, reports = [], []

    def collect(name, seconds, sensor, features, scores, predictions):
        all_scores.append(scores.assign(sensor=sensor, features=features))
        report = pd.DataFrame(classification_report(
            predictions.label, predictions.predicted, labels=LABELS,
            target_names=list(ACTIVITIES.values()), output_dict=True, zero_division=0)).T
        report.to_csv(out / f'{name}_{seconds}s_{sensor}_class_report.csv')
        table = report.loc[list(ACTIVITIES.values())].reset_index(names='activity')
        reports.append(table.assign(model=name, window_seconds=seconds, sensor=sensor))

    for seconds in durations:
        X, meta = datasets[seconds]
        for name in model_names:
            reference_scores, reference_predictions = references[name, seconds]
            collect(name, seconds, 'both', 58, reference_scores, reference_predictions)
            for sensor in sensors:
                selected = feature_columns(X, sensor)
                print(f'\n{name}, {seconds}s, {sensor}: {len(selected)} features', flush=True)
                operation = load_baseline if summary_only else run_baseline
                scores, predictions = operation(X[selected], meta, estimators[name],
                    name, seconds, output_dir=out / sensor)
                check_predictions(meta, scores, predictions)
                alignment = ['fold', 'test_subjects', 'train_windows', 'test_windows']
                if not scores[alignment].equals(reference_scores[alignment]):
                    raise ValueError('Single-sensor folds differ from the reference.')
                collect(name, seconds, sensor, len(selected), scores, predictions)
    scores = pd.concat(all_scores, ignore_index=True)
    scores['predict_ms_per_window'] = 1000 * scores.predict_seconds / scores.test_windows
    scores.to_csv(out / 'folds.csv', index=False)
    summary = scores.groupby(['model', 'window_seconds', 'sensor', 'features']).agg(
        accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'),
        f1_std=('macro_f1', 'std'), present_class_macro_f1=('present_class_macro_f1', 'mean'),
        folds=('fold', 'count'), fit_seconds=('fit_seconds', 'sum'),
        predict_ms_per_window=('predict_ms_per_window', 'mean')).reset_index()
    summary.to_csv(out / 'summary.csv', index=False)
    keys = ['model', 'window_seconds', 'fold', 'test_subjects']
    reference = scores[scores.sensor == 'both'][keys + ['macro_f1']].rename(
        columns={'macro_f1': 'both_macro_f1'})
    paired = scores[scores.sensor != 'both'].merge(reference, on=keys, validate='many_to_one')
    paired['delta'] = paired.macro_f1 - paired.both_macro_f1
    paired.to_csv(out / 'paired_vs_both.csv', index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for (name, seconds, sensor), group in paired.groupby(['model', 'window_seconds', 'sensor']):
        delta = group.delta.to_numpy()
        boot = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
        lo, hi = np.quantile(boot, [.025, .975])
        rows.append({'model': name, 'window_seconds': seconds, 'sensor': sensor,
                     'mean_delta': delta.mean(), 'ci_low': lo, 'ci_high': hi,
                     'subjects': len(delta)})
    differences = pd.DataFrame(rows)
    differences.to_csv(out / 'paired_difference_summary.csv', index=False)
    class_scores = pd.concat(reports, ignore_index=True)
    class_scores.to_csv(out / 'class_scores.csv', index=False)
    save_plots(summary, differences, class_scores, out)
    print('\n', summary.round(4).to_string(index=False), flush=True)
    print(f'\nSaved sensor ablation outputs: {out}', flush=True)
    return summary, differences

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', type=str.lower, choices=MODEL_NAMES,
                        default=['lr'])
    parser.add_argument('--windows', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--sensors', nargs='+', choices=['back', 'thigh'], default=['back', 'thigh'])
    parser.add_argument('--summary-only', action='store_true', help='Read completed ablation runs without fitting.')
    args = parser.parse_args(argv)
    run_ablation([MODEL_NAMES[x] for x in args.models], args.windows, args.sensors,
                 summary_only=args.summary_only)

if __name__ == '__main__':
    main()
