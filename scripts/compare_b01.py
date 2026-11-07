"""Compare frozen B01 checkpoints on full val and val excluding selection feeds."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import polars as pl
from neuroforge.metrics import ndcg


def cluster_interval(frame, seed=2026, repeats=2000):
    """Paired bootstrap over user clusters; interval is exploratory, not a guarantee."""
    clusters = frame.group_by('user_id').agg(pl.col('difference').sum(), pl.len().alias('n')).sort('user_id')
    sums = clusters['difference'].to_numpy()
    counts = clusters['n'].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        ids = rng.integers(0, len(sums), size=len(sums))
        estimates.append(float(sums[ids].sum() / counts[ids].sum()))
    return np.quantile(estimates, [.025, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = pl.read_parquet(sorted((args.data / 'val/impressions').glob('*.parquet')),
                             columns=['feed_id', 'item_id', 'target', 'user_id', 'timestamp'])
    meta = source.select('feed_id', 'user_id', 'timestamp').with_columns(
        (pl.col('timestamp').cast(pl.Int64) // 86400).alias('utc_day'))
    reference = source.select('feed_id', 'item_id', 'target').explode(['item_id', 'target'])
    del source
    all_rows, summary = [], {}
    for run in args.runs:
        selected = set(pl.read_parquet(run / 'predictions_val_ndcg.parquet', columns=['feed_id'])['feed_id'].to_list())
        model_frames = {}
        for criterion in ['logloss', 'ndcg']:
            prediction = pl.read_parquet(run / f'val-full-{criterion}/predictions.parquet')
            if len(prediction) != len(reference):
                raise ValueError('Coverage mismatch')
            joined = reference.join(prediction, on=['feed_id', 'item_id'], how='left', validate='1:1')
            if joined['score'].null_count() or not joined['score'].is_finite().all():
                raise ValueError('Missing/nonfinite prediction')
            rows = []
            for key, group in joined.group_by('feed_id', maintain_order=True):
                yy, ss = group['target'].to_list(), group['score'].to_list()
                n, positives = len(yy), sum(yy)
                ideal = sum(1 / np.log2(i + 2) for i in range(positives))
                expected_random = (positives / n * sum(1 / np.log2(i + 2) for i in range(n)) / ideal) if ideal else 1.
                rows.append({'feed_id': key[0], 'ndcg': ndcg(yy, ss),
                             'ndcg_at_10': ndcg(yy, ss, 10), 'ndcg_at_30': ndcg(yy, ss, 30),
                             'random_expected_ndcg': expected_random})
            frame = pl.DataFrame(rows).cast({'feed_id': pl.Int32}).join(meta, on='feed_id', validate='1:1').with_columns(
                pl.col('feed_id').is_in(list(selected)).alias('used_for_selection'))
            model_frames[criterion] = frame
            all_rows.append(frame.with_columns(pl.lit(run.name).alias('run'), pl.lit(criterion).alias('criterion')))
            print(f'{run.name}/{criterion}: evaluated {len(frame)} feeds', flush=True)
        paired = model_frames['ndcg'].join(model_frames['logloss'].select('feed_id', pl.col('ndcg').alias('control_ndcg')),
                                            on='feed_id', validate='1:1').with_columns(
            (pl.col('ndcg') - pl.col('control_ndcg')).alias('difference'))
        result = {'selected_checkpoints': json.loads((run / 'metrics.json').read_text())['selected']}
        for name, filter_expr in [('full_val', pl.lit(True)), ('selection_100', pl.col('used_for_selection')),
                                  ('remaining_val', ~pl.col('used_for_selection'))]:
            part = paired.filter(filter_expr)
            result[name] = {'feeds': len(part), 'logloss_selected_ndcg': part['control_ndcg'].mean(),
                            'ndcg_selected_ndcg': part['ndcg'].mean(), 'difference': part['difference'].mean(),
                            'random_expected_ndcg': part['random_expected_ndcg'].mean()}
            if name == 'remaining_val':
                result[name]['paired_user_bootstrap_95pct'] = cluster_interval(part)
                result[name]['by_utc_day'] = part.group_by('utc_day').agg(
                    pl.len().alias('feeds'), pl.col('ndcg').mean(), pl.col('control_ndcg').mean(),
                    pl.col('difference').mean()).sort('utc_day').to_dicts()
        summary[run.name] = result
    args.output.mkdir(parents=True)
    pl.concat(all_rows).write_parquet(args.output / 'per_feed_metrics.parquet')
    (args.output / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
