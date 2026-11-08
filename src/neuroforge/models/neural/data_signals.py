"""Only retained B02 signals, computed without labels or candidate order."""
import numpy as np
from .history_mean import vectors

NAMES = ('user_cos', 'recent_cos', 'top3_recent_cos')


def unit(value):
    return value / max(float(np.linalg.norm(value)), 1e-12)


def extract_signals(table, indices, items, user):
    values, present = vectors(table, indices[:32])
    usable = present & (np.linalg.norm(values, axis=1) > 0)
    recent = values[usable]
    mean = recent.mean(0) if len(recent) else np.zeros(128, dtype=np.float32)
    items = np.asarray(items, dtype=np.float32)
    user = np.asarray(user, dtype=np.float32)
    if not len(items):
        raise ValueError('Empty candidate set')
    matches = items @ recent.T
    top3 = np.sort(matches, axis=1)[:, -min(3, len(recent)):].mean(1) if len(recent) else np.zeros(len(items))
    return np.stack([items @ unit(user), items @ unit(mean), top3], axis=1).astype(np.float32)
