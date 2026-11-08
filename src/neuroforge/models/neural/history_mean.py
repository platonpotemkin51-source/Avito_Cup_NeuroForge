"""B02: bounded-shard input and mean-history extension of the organizer ranker."""
import importlib.util
import json
import mmap
from pathlib import Path
import random

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info

FLOATS = ['f1', 'f2', 'f3', 'f4', 'f5']
COLUMNS = ['feed_id', 'item_id', 'category_id', *FLOATS, 'item_id__embedding_idx',
           'user_embedding', 'history__item_id__embedding_idx']


def vectors(table, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < -1) or np.any(indices >= len(table)):
        raise ValueError('Invalid embedding index')
    present = indices >= 0
    result = np.zeros((len(indices), 128), dtype=np.float32)
    result[present] = table[indices[present]].astype(np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    np.divide(result, norms, out=result, where=norms > 0)
    return result, present


def mean_history(table, indices):
    values, present = vectors(table, indices)
    usable = present & (np.linalg.norm(values, axis=1) > 0)
    result = np.zeros(129, dtype=np.float32)
    if usable.any():
        result[:128] = values[usable].mean(axis=0)
    result[128] = float(not usable.any())  # explicit absence of usable history
    return torch.from_numpy(result)


class HistoryFeeds(IterableDataset):
    """Keep at most one Parquet shard in RAM, embeddings remain int8 memmap.

    Prefix limits are fixed before epoch shuffling; num_workers=0 is required.
    """
    def __init__(self, split, max_feeds=0, mapping=None, shuffle=False, seed=42, include_history=True, release_pages=False):
        super().__init__()
        self.split = Path(split)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.include_history, self.release_pages = include_history, release_pages
        if max_feeds < 0:
            raise ValueError('Nonnegative feed limit required')
        self.parts, self.count = [], 0
        remaining = max_feeds
        categories = set()
        for path in sorted((self.split / 'impressions').glob('*.parquet')):
            source = pl.scan_parquet(path)
            n = source.select(pl.len()).collect().item()
            take = min(n, remaining) if max_feeds else n
            if take:
                self.parts.append((path, take))
                self.count += take
                if mapping is None:
                    categories.update(source.head(take).select(pl.col('category_id').explode().unique()).collect().to_series().to_list())
            if max_feeds:
                remaining -= take
                if remaining == 0:
                    break
        if not self.count:
            raise ValueError('Empty feed selection')
        self.mapping = ({c: i + 1 for i, c in enumerate(sorted(categories - {0}))}
                        if mapping is None else mapping)
        meta = json.loads((self.split / 'item_embeddings/data.meta').read_text())
        if meta['dtype'] != 'int8' or meta['shape'][1] != 128:
            raise ValueError('Expected int8 128-dimensional embeddings')
        self.items = np.memmap(self.split / 'item_embeddings/data.memmap', mode='r',
                               dtype=meta['dtype'], shape=tuple(meta['shape']), order=meta['order'])

    def __len__(self):
        return self.count

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError('Use num_workers=0 to avoid duplicate feeds')
        rng = random.Random(self.seed + self.epoch)
        parts = list(self.parts)
        if self.shuffle:
            rng.shuffle(parts)
        for path, take in parts:
            schema = pl.scan_parquet(path).collect_schema()
            columns = [c for c in COLUMNS if self.include_history or c != 'history__item_id__embedding_idx'] + (['target'] if 'target' in schema else [])
            frame = pl.read_parquet(path, columns=columns, n_rows=take)
            order = list(range(len(frame)))
            if self.shuffle:
                rng.shuffle(order)
            for index in order:
                row = frame.row(index, named=True)
                count = len(row['item_id'])
                if count < 1 or any(len(row[k]) != count for k in ['category_id', *FLOATS, 'item_id__embedding_idx']):
                    raise ValueError('Candidate arrays must be aligned and nonempty')
                if 'target' in row and (len(row['target']) != count or any(y not in (0, 1) for y in row['target'])):
                    raise ValueError('Invalid targets')
                items, present = vectors(self.items, row['item_id__embedding_idx'])
                numeric = np.stack([row[k] for k in FLOATS], axis=1).astype(np.float32)
                features = torch.from_numpy(np.concatenate([numeric, items, (~present).astype(np.float32)[:, None]], axis=1))
                user = F.normalize(torch.tensor(row['user_embedding'], dtype=torch.float32), dim=0)
                yield {'features': features, 'category': torch.tensor([self.mapping.get(c, 0) for c in row['category_id']]),
                       'user': user, 'history': mean_history(self.items, row['history__item_id__embedding_idx']) if self.include_history else torch.zeros(129),
                       'target': torch.tensor(row['target'], dtype=torch.float32) if 'target' in row else None,
                       'feed_id': row['feed_id'], 'item_id': row['item_id']}
            del frame
            if self.release_pages and hasattr(self.items._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                try:
                    self.items._mmap.madvise(mmap.MADV_DONTNEED)
                except (OSError, ValueError):
                    pass  # OS may decline advice; it never changes the data.


def collate_history(rows):
    lengths = torch.tensor([len(r['item_id']) for r in rows])
    return {'features': pad_sequence([r['features'] for r in rows], batch_first=True),
            'category': pad_sequence([r['category'] for r in rows], batch_first=True),
            'users': torch.stack([r['user'] for r in rows]),
            'history': torch.stack([r['history'] for r in rows]),
            'padding': torch.arange(lengths.max())[None, :] >= lengths[:, None],
            'target': pad_sequence([r['target'] for r in rows], batch_first=True) if rows[0]['target'] is not None else None,
            'keys': [(r['feed_id'], item) for r in rows for item in r['item_id']]}


def make_ranker(category_count, use_history=True, width=64, layers=2, dropout=.1):
    path = Path(__file__).resolve().parents[4] / 'baseline/neural_baseline.py'
    spec = importlib.util.spec_from_file_location('b02_original', path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)

    class MeanHistoryRanker(baseline.Ranker):
        def __init__(self):
            super().__init__(category_count, width, layers, dropout)
            self.user = nn.Linear(257, width)
            self.use_history = use_history

        def forward(self, features, category, users, padding, history):
            history = history if self.use_history else torch.zeros_like(history)
            return super().forward(features, category, torch.cat([users, history], dim=1), padding)

    return MeanHistoryRanker()
