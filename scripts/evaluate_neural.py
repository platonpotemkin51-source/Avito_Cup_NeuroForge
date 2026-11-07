"""Evaluate the unchanged organizer neural baseline on an explicit development reference."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import polars as pl
import torch
from torch.utils.data import DataLoader
from neuroforge.metrics import ndcg


def load_baseline():
    spec = importlib.util.spec_from_file_location('organizer_neural', ROOT / 'baseline/neural_baseline.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(model, dataset, loader):
    """Keep whole-feed boundaries and key order; score exactly the exported float32 logits."""
    totals = {'ndcg': 0.0, 'ndcg_at_10': 0.0, 'ndcg_at_30': 0.0}
    scores = []
    feeds = 0
    model.eval()
    with torch.inference_mode():
        for features, categories, users, padding, targets in loader:
            if targets is None:
                raise ValueError('Evaluation requires targets')
            logits = model(features, categories, users, padding).float().cpu()
            for i in range(len(users)):
                real = ~padding[i]
                ss = logits[i][real].tolist()
                yy = targets[i][real].to(torch.int64).tolist()
                for name, cutoff in [('ndcg', None), ('ndcg_at_10', 10), ('ndcg_at_30', 30)]:
                    totals[name] += ndcg(yy, ss, cutoff)
                scores.extend(ss)
                feeds += 1
    if not feeds or len(scores) != len(dataset.keys):
        raise ValueError('Empty evaluation or prediction/key count mismatch')
    predictions = dataset.keys.with_columns(pl.Series('score', scores, dtype=pl.Float32)).cast(
        {'feed_id': pl.Int32, 'item_id': pl.Int32})
    return predictions, {**{k: v / feeds for k, v in totals.items()}, 'feeds': feeds, 'candidates': len(scores)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Competition root; uses its val directory')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--max-feeds', type=int, default=100, help='First whole feeds in sorted files; 0 means full supplied val')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.max_feeds < 0 or args.batch_size < 1 or args.threads < 1:
        parser.error('max-feeds must be nonnegative; batch-size and threads must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    baseline = load_baseline()
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    mapping = checkpoint.get('categories_mapping')
    if mapping is None:
        mapping = {value: i + 1 for i, value in enumerate(checkpoint['categories'])}
    model = baseline.Ranker(len(mapping), **checkpoint.get('model_params', {}))
    model.load_state_dict(checkpoint['state_dict'])
    dataset = baseline.CandidateData(args.data / 'val', args.max_feeds, mapping)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=baseline.collate_feeds)
    predictions, metrics = evaluate(model, dataset, loader)
    metrics.update(seconds=time.perf_counter() - started, checkpoint_epoch=checkpoint.get('epoch'),
                   model=str(args.model), data=str(args.data), max_feeds=args.max_feeds,
                   batch_size=args.batch_size, threads=args.threads, device='cpu', score='float32 logits',
                   selection='first whole feeds in lexicographically sorted parquet files; not representative sampling')
    args.output.mkdir(parents=True, exist_ok=False)
    predictions.write_parquet(args.output / 'predictions.parquet', compression='zstd')
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
