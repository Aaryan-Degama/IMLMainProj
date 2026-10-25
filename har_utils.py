"""Small, explicit helpers for the HARTH notebooks (50 Hz UCI release)."""
from pathlib import Path
import json
import platform
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
AXES = [f'{sensor}_{axis}' for sensor in ['back', 'thigh'] for axis in 'xyz']
ACTIVITIES = {1: 'walking', 2: 'running', 3: 'shuffling', 4: 'stairs_up',
              5: 'stairs_down', 6: 'standing', 7: 'sitting', 8: 'lying',
              13: 'cycling_sit', 14: 'cycling_stand',
              130: 'cycling_sit_inactive', 140: 'cycling_stand_inactive'}
LABELS = list(ACTIVITIES)
SEED = 42


def record_environment():
    """Record the active notebook/script environment, without installing packages."""
    import importlib.metadata

    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    packages = ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'xgboost',
                'nbformat', 'nbclient', 'ipykernel']
    versions = {name: importlib.metadata.version(name) for name in packages}
    (out / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'packages': versions}, indent=2))
    (ROOT / 'requirements-lock.txt').write_text(
        '\n'.join(f'{name}=={version}' for name, version in versions.items()) + '\n')


def subject_files():
    files = sorted((ROOT / 'data' / 'harth').glob('S*.csv'))
    if len(files) != 22:
        raise ValueError(f'Expected 22 HARTH subject CSVs in {ROOT / "data" / "harth"}, found {len(files)}.')
    return files


def read_subject(path):
    frame = pd.read_csv(path, usecols=['timestamp', *AXES, 'label'])
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    # Do not silently reorder, interpolate, or bridge missing data.
    if not frame['timestamp'].is_monotonic_increasing:
        raise ValueError(f'Unsorted timestamps in {path.name}')
    if not set(frame['label'].unique()).issubset(ACTIVITIES):
        raise ValueError(f'Unknown labels in {path.name}')
    return frame


def window_features(windows):
    """Input: [windows, samples, six axes]. FFT uses centered signals, excludes DC."""
    magnitudes = np.stack([np.linalg.norm(windows[:, :, :3], axis=2),
                           np.linalg.norm(windows[:, :, 3:], axis=2)], axis=2)
    signals = np.concatenate([windows, magnitudes], axis=2)
    channels = AXES + ['back_mag', 'thigh_mag']
    values = {}
    stats = {'mean': signals.mean(axis=1), 'std': signals.std(axis=1),
             'min': signals.min(axis=1), 'max': signals.max(axis=1),
             'median': np.median(signals, axis=1), 'range': np.ptp(signals, axis=1),
             'rms': np.sqrt((signals ** 2).mean(axis=1))}
    for name, array in stats.items():
        for i, channel in enumerate(channels):
            values[f'{channel}__{name}'] = array[:, i]
    for sensor, start in [('back', 0), ('thigh', 3)]:
        values[f'{sensor}__sma'] = np.abs(windows[:, :, start:start+3]).sum(axis=2).mean(axis=1)
    centered = signals - signals.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1))[:, 1:, :] ** 2
    frequencies = np.fft.rfftfreq(signals.shape[1], d=1 / 50)[1:]
    total = power.sum(axis=1)
    probability = power / np.maximum(total[:, None, :], 1e-12)
    dominant = frequencies[np.argmax(power, axis=1)]
    dominant[total < 1e-12] = 0
    centroid = (power * frequencies[None, :, None]).sum(axis=1) / np.maximum(total, 1e-12)
    entropy = -(probability * np.log2(np.maximum(probability, 1e-12))).sum(axis=1)
    for name, array in [('dominant_hz', dominant), ('centroid_hz', centroid), ('spectral_entropy', entropy)]:
        for i, channel in enumerate(channels):
            values[f'{channel}__{name}'] = array[:, i]
    return pd.DataFrame(values)


def build_windows(seconds=2):
    """Non-overlapping windows within subjects and uninterrupted timestamp runs.

    Majority voting labels windows AFTER segmentation; activity labels never define
    boundaries. Keep mixed windows and report their purity instead of hiding them.
    """
    size = int(seconds * 50)
    if size < 2 or size != seconds * 50:
        raise ValueError('Window duration must correspond to at least two whole samples.')
    features, metadata, audit = [], [], []
    for path in subject_files():
        frame = read_subject(path)
        dt = frame.timestamp.diff().dt.total_seconds().to_numpy()
        valid = np.isfinite(frame[AXES].to_numpy()).all(axis=1)
        # Split at invalid rows and at any departure from the nominal 20 ms cadence.
        breaks = np.r_[0, np.flatnonzero((dt[1:] < .019) | (dt[1:] > .021)
                                        | ~valid[1:] | ~valid[:-1]) + 1, len(frame)]
        kept = 0
        for left, right in zip(breaks[:-1], breaks[1:]):
            n = (right - left) // size
            if n == 0 or not valid[left:right].all():
                continue
            stop = left + n * size
            raw = frame.iloc[left:stop][AXES].to_numpy().reshape(n, size, 6)
            labels = frame.label.iloc[left:stop].to_numpy().reshape(n, size)
            votes = np.stack([(labels == label).sum(axis=1) for label in LABELS], axis=1)
            winner = votes.argmax(axis=1)  # Ties use the first code in LABELS.
            features.append(window_features(raw))
            starts = left + np.arange(n) * size
            metadata.append(pd.DataFrame({'subject': path.stem, 'start_row': starts,
                'start_time': frame.timestamp.iloc[starts].to_numpy(),
                'label': np.array(LABELS)[winner], 'purity': votes.max(axis=1) / size}))
            kept += n * size
        audit.append({'subject': path.stem, 'input_samples': len(frame),
                      'used_samples': kept, 'discarded_samples': len(frame) - kept})
    X = pd.concat(features, ignore_index=True)
    meta = pd.concat(metadata, ignore_index=True)
    assert len(X) == len(meta) and np.isfinite(X.to_numpy()).all()
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    X.to_pickle(out / f'features_{seconds}s.pkl')
    meta.to_csv(out / f'windows_{seconds}s.csv', index=False)
    pd.DataFrame(audit).to_csv(out / f'window_audit_{seconds}s.csv', index=False)
    return X, meta


def load_windows(seconds=2):
    for filename in [f'features_{seconds}s.pkl', f'windows_{seconds}s.csv']:
        if not (ROOT / 'results' / filename).exists():
            raise FileNotFoundError(
                f'Missing {filename}. Run notebook 02 for 2-second features; '
                'notebook 05 prepares 4-second features.')
    return (pd.read_pickle(ROOT / 'results' / f'features_{seconds}s.pkl'),
            pd.read_csv(ROOT / 'results' / f'windows_{seconds}s.csv'))


def feature_columns(X, representation='time'):
    frequency = ['dominant_hz', 'centroid_hz', 'spectral_entropy']
    time_cols = [c for c in X if c.split('__')[-1] not in frequency]
    if representation == 'time':
        return time_cols
    if representation == 'time_frequency':
        return list(X.columns)
    if representation in ['back', 'thigh']:
        return [c for c in time_cols if c.startswith(representation + '_')]
    if representation == 'magnitude':
        return [c for c in time_cols if '_mag__' in c]
    raise ValueError(representation)


def models():
    # Fixed settings: no test-subject tuning. Linear SVM keeps full LOSO tractable.
    return {
        'Dummy': DummyClassifier(strategy='most_frequent'),
        'LR': make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1500, random_state=SEED)),
        'LinearSVM': make_pipeline(StandardScaler(), LinearSVC(C=.1, dual=False, max_iter=5000, random_state=SEED)),
        'RF': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=2,
                                     n_jobs=4, random_state=SEED),
        'XGBoost': XGBClassifier(n_estimators=150, max_depth=4, learning_rate=.1,
                                subsample=.8, colsample_bytree=.8, tree_method='hist',
                                n_jobs=4, random_state=SEED),
    }


def evaluate(X, meta, estimator, name, protocol='loso'):
    """Save every held-out prediction, fold metrics, and parameters for review."""
    y = meta.label.to_numpy()
    groups = meta.subject.to_numpy()
    if protocol == 'loso':
        splits = LeaveOneGroupOut().split(X, y, groups)
    elif protocol == 'group4':
        splits = GroupKFold(4).split(X, y, groups)
    elif protocol == 'random':
        splits = StratifiedShuffleSplit(1, test_size=.25, random_state=SEED).split(X, y)
    else:
        raise ValueError(protocol)
    rows, predictions = [], []
    for fold, (train, test) in enumerate(splits):
        if protocol != 'random':
            assert not set(groups[train]) & set(groups[test])
        encoder = LabelEncoder().fit(y[train])
        model = clone(estimator)
        start = time.perf_counter()
        model.fit(X.iloc[train], encoder.transform(y[train]))
        pred = encoder.inverse_transform(model.predict(X.iloc[test]).astype(int))
        rows.append({'experiment': name, 'protocol': protocol, 'fold': fold,
            'test_subjects': ','.join(sorted(set(groups[test]))),
            'train_windows': len(train), 'test_windows': len(test),
            'accuracy': accuracy_score(y[test], pred),
            # Fixed 12-class macro F1, including absent classes as zero.
            'macro_f1': f1_score(y[test], pred, labels=LABELS, average='macro', zero_division=0),
            'present_class_macro_f1': f1_score(y[test], pred, labels=np.unique(y[test]), average='macro', zero_division=0),
            'seconds': time.perf_counter() - start})
        output = meta.iloc[test].copy()
        output['window_id'] = test
        output['predicted'] = pred
        output['fold'] = fold
        predictions.append(output)
        print(f'{name} {protocol} fold {fold+1}: F1={rows[-1]["macro_f1"]:.3f}', flush=True)
    scores = pd.DataFrame(rows)
    predictions = pd.concat(predictions, ignore_index=True)
    out = ROOT / 'results'
    scores.to_csv(out / f'{name}_{protocol}_folds.csv', index=False)
    predictions.to_csv(out / f'{name}_{protocol}_predictions.csv', index=False)
    config = {'name': name, 'protocol': protocol, 'seed': SEED,
              'feature_columns': list(X.columns), 'parameters': estimator.get_params(deep=True)}
    (out / f'{name}_{protocol}_config.json').write_text(json.dumps(config, default=str, indent=2))
    return scores, predictions


def summarize(scores):
    return scores.groupby(['experiment', 'protocol']).agg(
        accuracy=('accuracy', 'mean'), macro_f1=('macro_f1', 'mean'),
        f1_std=('macro_f1', 'std'), present_class_macro_f1=('present_class_macro_f1', 'mean'),
        folds=('fold', 'count'), seconds=('seconds', 'sum')).reset_index()


def save_summary():
    """Refresh the summary from all completed experiment fold files."""
    files = sorted((ROOT / 'results').glob('*_folds.csv'))
    if files:
        table = summarize(pd.concat([pd.read_csv(path) for path in files]))
        table.to_csv(ROOT / 'results' / 'summary.csv', index=False)
        return table