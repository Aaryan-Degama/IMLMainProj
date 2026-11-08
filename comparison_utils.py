"""HARTH evaluation and learning-curve utilities."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from har_utils import (ROOT, SEED, LABELS, build_windows, load_windows,
                       feature_columns, models)

OUT = ROOT / 'results' / 'window_comparison'

def comparison_models(workers=2):
    """Fixed baseline estimators with a configurable worker count."""
    estimators = models()
    for name in ['RF', 'XGBoost']:
        estimators[name].set_params(n_jobs=workers)
    return {name: estimators[name] for name in ['LR', 'RF', 'XGBoost']}

def prepare_windows(seconds, rebuild=False):
    """Load or build cached window features."""
    paths = [ROOT / 'results' / f'features_{seconds}s.pkl',
             ROOT / 'results' / f'windows_{seconds}s.csv']
    if rebuild or not all(p.exists() for p in paths):
        return build_windows(seconds)
    return load_windows(seconds)

def validate_windows(X, meta):
    if len(X) != len(meta) or not len(X):
        raise ValueError('Feature/metadata lengths differ or are empty; rebuild windows.')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Non-finite features; rebuild windows.')
    if not set(meta.label).issubset(LABELS):
        raise ValueError('Unknown activity labels.')
    if meta.duplicated(['subject', 'start_row']).any():
        raise ValueError('Duplicate window identities.')

def fingerprint(X, meta, config):
    """Invalidate checkpoints on changed features, labels, settings or packages."""
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(X, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(meta, index=True).values.tobytes())
    versions = {p: importlib.metadata.version(p) for p in
                ['numpy', 'pandas', 'scikit-learn', 'xgboost']}
    digest.update(json.dumps({'columns': list(X), 'config': config,
                              'versions': versions, 'schema': 1},
                             sort_keys=True, default=str).encode())
    return digest.hexdigest()

def metrics(y, predicted):
    return {'accuracy': accuracy_score(y, predicted),
            'macro_f1': f1_score(y, predicted, labels=LABELS,
                                 average='macro', zero_division=0),
            'present_class_macro_f1': f1_score(
                y, predicted, labels=np.unique(y), average='macro', zero_division=0)}

def run_baseline(X, meta, estimator, name, seconds, force=False, output_dir=None):
    """Full LOSO, one resumable checkpoint per model/duration/test subject."""
    validate_windows(X, meta)
    config = {'model': name, 'seconds': seconds, 'protocol': 'loso',
              'parameters': estimator.get_params(deep=True), 'seed': SEED,
              'features': list(X), 'labels': LABELS}
    key = fingerprint(X, meta, config)
    out = OUT if output_dir is None else Path(output_dir)
    folder = out / f'{name}_{seconds}s' / key
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    y, groups = meta.label.to_numpy(), meta.subject.to_numpy()
    rows, predictions = [], []
    splits = list(LeaveOneGroupOut().split(X, y, groups))
    for fold, (train, test) in enumerate(splits):
        score_path = folder / f'fold_{fold:02d}.json'
        pred_path = folder / f'fold_{fold:02d}_predictions.csv'
        if not force and score_path.exists() and pred_path.exists():
            row = json.loads(score_path.read_text())
            prediction = pd.read_csv(pred_path)
            print(f'{name} {seconds}s {fold+1}/{len(splits)}: cached', flush=True)
        else:
            assert not set(groups[train]) & set(groups[test])
            encoder = LabelEncoder().fit(y[train])
            model = clone(estimator)
            started = time.perf_counter()
            model.fit(X.iloc[train], encoder.transform(y[train]))
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predicted = encoder.inverse_transform(model.predict(X.iloc[test]).astype(int))
            predict_seconds = time.perf_counter() - started
            row = {'model': name, 'window_seconds': seconds, 'fold': fold,
                   'test_subjects': ','.join(sorted(set(groups[test]))),
                   'train_windows': len(train), 'test_windows': len(test),
                   'fit_seconds': fit_seconds, 'predict_seconds': predict_seconds,
                   **metrics(y[test], predicted)}
            prediction = meta.iloc[test].copy()
            prediction['window_id'] = test
            prediction['predicted'] = predicted
            prediction['fold'] = fold
            prediction.to_csv(pred_path, index=False)

            score_path.write_text(json.dumps(row, indent=2))
            print(f'{name} {seconds}s {fold+1}/{len(splits)}: '
                  f'F1={row["macro_f1"]:.3f}', flush=True)
        rows.append(row)
        predictions.append(prediction)
    scores = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    scores.to_csv(folder / 'folds.csv', index=False)
    predictions.to_csv(folder / 'predictions.csv', index=False)

    (out / f'{name}_{seconds}s_latest.json').write_text(json.dumps({
        'fingerprint': key, 'folder': str(folder.relative_to(out)),
        'expected_folds': len(splits)}, indent=2))
    return scores, predictions

def load_baseline(X, meta, estimator, name, seconds, output_dir=None):
    config = {'model': name, 'seconds': seconds, 'protocol': 'loso',
              'parameters': estimator.get_params(deep=True), 'seed': SEED,
              'features': list(X), 'labels': LABELS}
    key = fingerprint(X, meta, config)
    out = OUT if output_dir is None else Path(output_dir)
    folder = out / f'{name}_{seconds}s' / key
    if not (folder / 'folds.csv').exists():
        raise FileNotFoundError(f'Missing matching LOSO results: {name}, {seconds}s.')
    scores = pd.read_csv(folder / 'folds.csv')
    predictions = pd.read_csv(folder / 'predictions.csv')
    if scores.test_subjects.nunique() != meta.subject.nunique() or len(predictions) != len(meta):
        raise ValueError('Incomplete LOSO results.')
    return scores, predictions

def neural_diagnostic(datasets, epochs=30, batch_size=256, force=False):
    """MLP learning curves with subject-disjoint validation and test sets."""
    subjects = sorted(datasets[1][1].subject.unique())
    if len(subjects) < 10:
        raise ValueError('Need at least 10 subjects for this diagnostic.')
    split = {'test': subjects[:4], 'validation': subjects[4:8], 'train': subjects[8:]}
    classes = np.arange(len(LABELS))
    encode = {label: i for i, label in enumerate(LABELS)}
    outputs = {}
    for seconds, (X, meta) in datasets.items():
        X = X[feature_columns(X)]
        validate_windows(X, meta)
        assert sorted(meta.subject.unique()) == subjects
        settings = {'split': split, 'epochs': epochs, 'batch_size': batch_size,
                    'hidden_layer_sizes': [64, 32], 'learning_rate_init': .001,
                    'alpha': .0001, 'seed': SEED, 'labels': LABELS,
                    'selection': 'minimum validation log loss', 'schema': 1}
        key = fingerprint(X, meta, settings)
        folder = OUT / f'MLP_diagnostic_{seconds}s' / key
        folder.mkdir(parents=True, exist_ok=True)
        complete = folder / 'complete.json'
        if force or not complete.exists():
            masks = {role: meta.subject.isin(ids).to_numpy() for role, ids in split.items()}
            scaler = StandardScaler().fit(X.loc[masks['train']])
            arrays = {role: scaler.transform(X.loc[mask]) for role, mask in masks.items()}
            targets = {role: meta.loc[mask, 'label'].map(encode).to_numpy()
                       for role, mask in masks.items()}
            model = MLPClassifier(hidden_layer_sizes=(64, 32), solver='adam',
                                  alpha=.0001, learning_rate_init=.001,
                                  batch_size=batch_size, random_state=SEED,
                                  early_stopping=False, shuffle=False)
            rng = np.random.default_rng(SEED)
            history, best_loss, best_model, best_epoch = [], np.inf, None, None
            started = time.perf_counter()
            for epoch in range(1, epochs + 1):
                order = rng.permutation(len(targets['train']))
                for left in range(0, len(order), batch_size):
                    idx = order[left:left + batch_size]
                    model.set_params(batch_size=len(idx))
                    model.partial_fit(arrays['train'][idx], targets['train'][idx], classes=classes)
                row = {'epoch': epoch, 'window_seconds': seconds}
                for role in ['train', 'validation']:
                    proba = model.predict_proba(arrays[role])
                    predicted = proba.argmax(axis=1)
                    row[f'{role}_loss'] = log_loss(targets[role], proba, labels=classes)
                    row[f'{role}_accuracy'] = accuracy_score(targets[role], predicted)
                    row[f'{role}_macro_f1'] = f1_score(
                        targets[role], predicted, labels=classes, average='macro', zero_division=0)
                history.append(row)
                if row['validation_loss'] < best_loss:
                    best_loss, best_model, best_epoch = row['validation_loss'], deepcopy(model), epoch
                print(f'MLP {seconds}s epoch {epoch}/{epochs}: '
                      f'train={row["train_loss"]:.4f}, val={row["validation_loss"]:.4f}', flush=True)
            fit_seconds = time.perf_counter() - started
            predicted = np.array(LABELS)[best_model.predict(arrays['test']).astype(int)]
            prediction = meta.loc[masks['test']].copy()
            prediction['predicted'] = predicted
            row = {'window_seconds': seconds, 'best_epoch': best_epoch,
                   'validation_loss': best_loss, 'fit_seconds': fit_seconds,
                   **metrics(prediction.label, predicted)}
            pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
            prediction.to_csv(folder / 'predictions.csv', index=False)
            (folder / 'test_metrics.json').write_text(json.dumps(row, indent=2))
            (folder / 'config.json').write_text(json.dumps(settings, indent=2))
            complete.write_text(json.dumps({'fingerprint': key}))
        else:
            print(f'MLP {seconds}s: cached completed diagnostic', flush=True)
        outputs[seconds] = (pd.read_csv(folder / 'history.csv'),
                            json.loads((folder / 'test_metrics.json').read_text()),
                            pd.read_csv(folder / 'predictions.csv'))
    return outputs, split

def boosting_diagnostic(datasets, force=False):
    """Train/validation loss by boosting round, using the same split as the MLP."""
    from xgboost import DMatrix, train

    subjects = sorted(datasets[1][1].subject.unique())
    if len(subjects) < 10:
        raise ValueError('Need at least 10 subjects for this diagnostic.')
    split = {'test': subjects[:4], 'validation': subjects[4:8], 'train': subjects[8:]}
    encode = {label: i for i, label in enumerate(LABELS)}
    estimator = comparison_models()['XGBoost']
    rounds = estimator.n_estimators
    params = estimator.get_xgb_params()
    params.update(objective='multi:softprob', num_class=len(LABELS), eval_metric='mlogloss')
    outputs = {}
    for seconds, (X, meta) in datasets.items():
        X = X[feature_columns(X)]
        validate_windows(X, meta)
        assert sorted(meta.subject.unique()) == subjects
        config = {'split': split, 'parameters': params, 'rounds': rounds,
                  'labels': LABELS, 'selection': 'minimum validation log loss'}
        folder = OUT / f'XGBoost_diagnostic_{seconds}s' / fingerprint(X, meta, config)
        folder.mkdir(parents=True, exist_ok=True)
        complete = folder / 'complete.json'
        if force or not complete.exists():
            masks = {role: meta.subject.isin(ids).to_numpy() for role, ids in split.items()}
            matrices = {role: DMatrix(X.loc[masks[role]],
                        label=meta.loc[masks[role], 'label'].map(encode).to_numpy(), nthread=2)
                        for role in ['train', 'validation']}
            evaluations = {}
            started = time.perf_counter()
            booster = train(params, matrices['train'], num_boost_round=rounds,
                            evals=[(matrices['train'], 'train'),
                                   (matrices['validation'], 'validation')],
                            evals_result=evaluations, verbose_eval=25)
            fit_seconds = time.perf_counter() - started
            best_round = int(np.argmin(evaluations['validation']['mlogloss'])) + 1
            prediction = meta.loc[masks['test']].copy()
            test_matrix = DMatrix(X.loc[masks['test']], nthread=2)
            proba = booster.predict(test_matrix, iteration_range=(0, best_round))
            prediction['predicted'] = np.array(LABELS)[proba.argmax(axis=1)]
            history = pd.DataFrame({'round': np.arange(1, rounds + 1),
                'window_seconds': seconds, 'train_loss': evaluations['train']['mlogloss'],
                'validation_loss': evaluations['validation']['mlogloss']})
            result = {'window_seconds': seconds, 'best_round': best_round,
                      'validation_loss': min(evaluations['validation']['mlogloss']),
                      'fit_seconds': fit_seconds, **metrics(prediction.label, prediction.predicted)}
            history.to_csv(folder / 'history.csv', index=False)
            prediction.to_csv(folder / 'predictions.csv', index=False)
            (folder / 'test_metrics.json').write_text(json.dumps(result, indent=2))
            (folder / 'config.json').write_text(json.dumps(config, default=str, indent=2))
            complete.write_text(json.dumps({'complete': True}))
        else:
            print(f'XGBoost {seconds}s: cached completed diagnostic', flush=True)
        outputs[seconds] = (pd.read_csv(folder / 'history.csv'),
                            json.loads((folder / 'test_metrics.json').read_text()),
                            pd.read_csv(folder / 'predictions.csv'))
    return outputs, split
