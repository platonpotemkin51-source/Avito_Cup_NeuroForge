"""Bounded-shard data input and retained recent-history user token for B02."""
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
COLUMNS = ['feed_id', 'item_id', 'category_id', *FLOATS, 'item_id__embedding_idx', 'user_embedding', 'history__item_id__embedding_idx']


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
    result[128] = float(not usable.any())
    return torch.from_numpy(result)


class HistoryFeeds(IterableDataset):
    """One decoded shard at a time; global int8 vectors remain read-only memmap.

    max_feeds=0 selects all feeds. Limits are fixed before deterministic shuffle.
    Targets are never read when with_labels=False. Signals use available inputs only.
    """
    def __init__(self, split, max_feeds=0, mapping=None, shuffle=False, seed=42,
                 include_history=True, release_pages=True, history_mode='recent32',
                 with_signals=False, with_labels=True, pad_to=None):
        super().__init__()
        self.split = Path(split)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        self.include_history, self.release_pages = include_history, release_pages
        self.history_mode, self.with_signals = history_mode, with_signals
        self.with_labels, self.pad_to = with_labels, pad_to
        if max_feeds < 0 or history_mode != 'recent32':
            raise ValueError('Invalid feed limit/history mode')
        self.parts, self.count = [], 0
        remaining = max_feeds
        categories = set()
        for path in sorted((self.split / 'impressions').glob('*.parquet')):
            source = pl.scan_parquet(path)
            n = source.select(pl.len()).collect().item()
            take = min(n, remaining) if max_feeds else n
            if take:
                self.parts.append((path, take)); self.count += take
                if mapping is None:
                    categories.update(source.head(take).select(pl.col('category_id').explode().unique()).collect().to_series().to_list())
            if max_feeds:
                remaining -= take
                if remaining == 0: break
        if not self.count: raise ValueError('Empty feed selection')
        if max_feeds and remaining: raise ValueError('Requested more feeds than available')
        self.mapping = ({c: i+1 for i,c in enumerate(sorted(categories - {0}))} if mapping is None else mapping)
        meta = json.loads((self.split / 'item_embeddings/data.meta').read_text())
        if meta['dtype'] != 'int8' or meta['shape'][1] != 128:
            raise ValueError('Expected int8 128-dimensional embeddings')
        self.items = np.memmap(self.split / 'item_embeddings/data.memmap', mode='r', dtype=meta['dtype'], shape=tuple(meta['shape']), order=meta['order'])

    def __len__(self): return self.count

    def __iter__(self):
        if get_worker_info() is not None: raise RuntimeError('Use num_workers=0')
        rng = random.Random(self.seed + self.epoch)
        parts = list(self.parts)
        if self.shuffle: rng.shuffle(parts)
        for path,take in parts:
            schema = pl.scan_parquet(path).collect_schema()
            labels = self.with_labels and 'target' in schema
            frame = pl.read_parquet(path, columns=COLUMNS + (['target'] if labels else []), n_rows=take)
            order = list(range(len(frame)))
            if self.shuffle: rng.shuffle(order)
            for index in order:
                row = frame.row(index, named=True); count = len(row['item_id'])
                if count < 1 or any(len(row[k]) != count for k in ['category_id', *FLOATS, 'item_id__embedding_idx']):
                    raise ValueError('Candidate arrays must be aligned and nonempty')
                if labels and (len(row['target']) != count or any(y not in (0,1) for y in row['target'])):
                    raise ValueError('Invalid targets')
                items,present = vectors(self.items, row['item_id__embedding_idx'])
                numeric = np.stack([row[k] for k in FLOATS], axis=1).astype(np.float32)
                features = torch.from_numpy(np.concatenate([numeric, items, (~present).astype(np.float32)[:,None]], axis=1))
                user = F.normalize(torch.tensor(row['user_embedding'], dtype=torch.float32), dim=0)
                indices = row['history__item_id__embedding_idx']
                history = mean_history(self.items, indices[:32]) if self.include_history else torch.zeros(129)
                sample = {'features':features, 'category':torch.tensor([self.mapping.get(c,0) for c in row['category_id']]),
                          'user':user, 'history':history, 'target':torch.tensor(row['target'], dtype=torch.float32) if labels else None,
                          'feed_id':row['feed_id'], 'item_id':row['item_id']}
                if self.with_signals:
                    from .data_signals import extract_signals
                    sample['signals'] = torch.from_numpy(extract_signals(self.items, indices, items, user.numpy()))
                if self.pad_to is not None: sample['pad_to'] = self.pad_to
                yield sample
            del frame
            if self.release_pages and hasattr(self.items._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
                try: self.items._mmap.madvise(mmap.MADV_DONTNEED)
                except (OSError, ValueError): pass


def collate_history(rows):
    lengths = torch.tensor([len(r['item_id']) for r in rows])
    result = {'features':pad_sequence([r['features'] for r in rows], batch_first=True),
              'category':pad_sequence([r['category'] for r in rows], batch_first=True),
              'users':torch.stack([r['user'] for r in rows]), 'history':torch.stack([r['history'] for r in rows]),
              'padding':torch.arange(lengths.max())[None,:] >= lengths[:,None],
              'target':pad_sequence([r['target'] for r in rows], batch_first=True) if rows[0]['target'] is not None else None,
              'keys':[(r['feed_id'], item) for r in rows for item in r['item_id']]}
    if 'signals' in rows[0]: result['signals'] = pad_sequence([r['signals'] for r in rows], batch_first=True)
    pad_to = rows[0].get('pad_to')
    if any(row.get('pad_to') != pad_to for row in rows): raise ValueError('Mixed fixed-padding settings')
    if pad_to is not None:
        extra = int(pad_to) - result['padding'].shape[1]
        if extra < 0: raise ValueError('Feed exceeds fixed candidate padding')
        for key in ['features', 'signals']:
            if key in result: result[key] = F.pad(result[key], (0,0,0,extra))
        for key in ['category', 'target']:
            if result[key] is not None: result[key] = F.pad(result[key], (0,extra))
        result['padding'] = F.pad(result['padding'], (0,extra), value=True)
    return result


def make_ranker(category_count, use_history=True, width=64, layers=2, dropout=.1):
    path = Path(__file__).resolve().parents[4] / 'baseline/neural_baseline.py'
    spec = importlib.util.spec_from_file_location('b02_original', path)
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    class MeanHistoryRanker(baseline.Ranker):
        def __init__(self):
            super().__init__(category_count, width, layers, dropout)
            self.user = nn.Linear(257, width); self.use_history = use_history
        def forward(self, features, category, users, padding, history):
            history = history if self.use_history else torch.zeros_like(history)
            return super().forward(features, category, torch.cat([users, history], dim=1), padding)
    return MeanHistoryRanker()
