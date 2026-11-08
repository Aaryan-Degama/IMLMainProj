"""Subject-count learning curves for logistic regression and random forest."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import log_loss

from har_utils import ROOT, LABELS, SEED, load_windows, feature_columns
from comparison_utils import comparison_models, validate_windows, fingerprint, metrics

OUT = ROOT / 'results' / 'learning_curves'
MODEL_NAMES = {'lr': 'LR', 'rf': 'RF'}

def save_plots(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    names = list(summary.model.unique())
    durations = sorted(summary.window_seconds.unique())
    for metric, ylabel, filename in [
        ('macro_f1', 'Fixed-12-class macro F1', 'classical_learning_curves'),
        ('accuracy', 'Accuracy', 'classical_accuracy_vs_training_subjects'),
        ('loss', 'Cross-entropy', 'classical_loss_vs_training_subjects')]:
        fig, axes = plt.subplots(len(names), len(durations), squeeze=False,
                                 figsize=(6 * len(durations), 4 * len(names)))
        for row, name in enumerate(names):
            for col, seconds in enumerate(durations):
                ax = axes[row, col]
                part = summary[(summary.model == name) & (summary.window_seconds == seconds)]
                x = part.train_subject_count.to_numpy()
                for role, color in [('train', 'tab:blue'), ('validation', 'tab:orange')]:
                    mean = part[f'{role}_{metric}_mean'].to_numpy()
                    std = part[f'{role}_{metric}_std'].fillna(0).to_numpy()
                    ax.plot(x, mean, 'o-', color=color, label=role)
                    ax.fill_between(x, np.maximum(0, mean - std), mean + std,
                                    color=color, alpha=.15)
                ax.set(title=f'{name}, {seconds}s windows', xlabel='Training subjects', ylabel=ylabel)
                ax.set_xticks(x)
                if metric != 'loss':
                    ax.set_ylim(0, 1)
                ax.grid(alpha=.2)
                ax.legend()
        fig.tight_layout()
        for extension in ['png', 'pdf']:
            fig.savefig(figures / f'{filename}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1, len(durations), squeeze=False, figsize=(6 * len(durations), 4))
    for ax, seconds in zip(axes[0], durations):
        for name in names:
            part = summary[(summary.model == name) & (summary.window_seconds == seconds)]
            ax.plot(part.train_subject_count, part.validation_macro_f1_mean, 'o-', label=name)
        ax.set(title=f'{seconds}s windows', xlabel='Training subjects',
               ylabel='Validation macro F1', ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    for extension in ['png', 'pdf']:
        fig.savefig(figures / f'validation_model_comparison.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def run_curves(model_names=('LR', 'RF'), durations=(1, 2), sizes=(2, 4, 7, 10, 13),
               repeats=3, output_dir=None):
    names = tuple(dict.fromkeys(model_names))
    durations = tuple(dict.fromkeys(durations))
    sizes = sorted(set(sizes))
    if not names or not set(names).issubset(MODEL_NAMES.values()):
        raise ValueError('Unsupported model selection.')
    if not durations or not set(durations).issubset({1, 2}):
        raise ValueError('Unsupported window duration.')
    if not sizes or sizes[0] < 1 or repeats < 1:
        raise ValueError('Invalid learning-curve settings.')
    out = OUT if output_dir is None else Path(output_dir)
    estimators = comparison_models()
    datasets = {s: load_windows(s) for s in durations}
    subjects = sorted(next(iter(datasets.values()))[1].subject.unique())
    if len(subjects) < 10:
        raise ValueError('Insufficient subjects for the fixed split.')
    split = {'test': subjects[:4], 'validation': subjects[4:8], 'train': subjects[8:]}
    if sizes[-1] > len(split['train']):
        raise ValueError('Training subject count exceeds the fixed training pool.')
    orders = [np.random.default_rng(SEED + repeat).permutation(split['train']).tolist()
              for repeat in range(repeats)]
    for X, meta in datasets.values():
        validate_windows(X, meta)
        if sorted(meta.subject.unique()) != subjects:
            raise ValueError('Subject coverage differs across window durations.')
    out.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seconds, (X, meta) in datasets.items():
        X = X[feature_columns(X)]
        validation_mask = meta.subject.isin(split['validation']).to_numpy()
        eligible = meta.subject.isin(split['train'] + split['validation']).to_numpy()
        X_validation = X.loc[validation_mask]
        y_validation = meta.loc[validation_mask, 'label'].to_numpy()
        for name in names:
            config = {'model': name, 'window_seconds': seconds, 'split': split,
                      'features': list(X), 'parameters': estimators[name].get_params(deep=True),
                      'subject_orders': orders, 'sizes': sizes, 'repeats': repeats,
                      'seed': SEED, 'labels': LABELS, 'schema': 1}
            key = fingerprint(X.loc[eligible], meta.loc[eligible], config)
            folder = out / f'{name}_{seconds}s' / key
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'config.json').write_text(json.dumps(config, default=str, indent=2))
            for repeat, order in enumerate(orders):
                for count in sizes:
                    selected = sorted(order[:count])
                    train_mask = meta.subject.isin(selected).to_numpy()
                    if set(selected) & set(split['validation'] + split['test']):
                        raise ValueError('Subject sets overlap.')
                    subset = '-'.join(selected)
                    path = folder / f'n{count:02d}_{subset}.json'
                    if path.exists():
                        row = json.loads(path.read_text())
                        status = 'cached'
                    else:
                        X_train = X.loc[train_mask]
                        y_train = meta.loc[train_mask, 'label'].to_numpy()
                        if len(np.unique(y_train)) < 2:
                            raise ValueError('Training subset contains fewer than two activities.')
                        model = clone(estimators[name])
                        started = time.perf_counter()
                        model.fit(X_train, y_train)
                        fit_seconds = time.perf_counter() - started
                        row = {'model': name, 'window_seconds': seconds,
                               'train_subject_count': count, 'train_subjects': ','.join(selected),
                               'train_windows': len(y_train), 'validation_windows': len(y_validation),
                               'train_class_count': len(np.unique(y_train)),
                               'validation_class_count': len(np.unique(y_validation)),
                               'fit_seconds': fit_seconds}
                        for role, inputs, y in [('train', X_train, y_train),
                                                ('validation', X_validation, y_validation)]:
                            probability = model.predict_proba(inputs)
                            classes = model.classes_
                            predicted = classes[probability.argmax(axis=1)]
                            aligned = np.zeros((len(y), len(LABELS)))
                            for column, label in enumerate(classes):
                                aligned[:, LABELS.index(int(label))] = probability[:, column]
                            row.update({f'{role}_{metric}': value
                                        for metric, value in metrics(y, predicted).items()})
                            row[f'{role}_loss'] = log_loss(y, aligned, labels=LABELS)
                        temporary = path.with_suffix('.tmp')
                        temporary.write_text(json.dumps(row, indent=2))
                        temporary.replace(path)
                        status = 'fitted'
                    all_rows.append({**row, 'repeat': repeat})
                    print(f'{name} {seconds}s repeat {repeat+1}/{repeats}, subjects {count}: '
                          f'train F1={row["train_macro_f1"]:.3f}, '
                          f'validation F1={row["validation_macro_f1"]:.3f} ({status})', flush=True)
    points = pd.DataFrame(all_rows)
    points.to_csv(out / 'learning_curve_points.csv', index=False)
    metric_columns = [f'{role}_{metric}' for role in ['train', 'validation']
                      for metric in ['macro_f1', 'present_class_macro_f1', 'accuracy', 'loss']]
    summary = points.groupby(['model', 'window_seconds', 'train_subject_count'])[metric_columns].agg(
        ['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / 'learning_curve_summary.csv', index=False)
    save_plots(summary, out)
    print(f'Saved learning curves: {out}', flush=True)
    return points, summary

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--windows', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--sizes', nargs='+', type=int, default=[2, 4, 7, 10, 13])
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    run_curves([MODEL_NAMES[name] for name in args.models], args.windows, args.sizes, args.repeats)

if __name__ == '__main__':
    main()
