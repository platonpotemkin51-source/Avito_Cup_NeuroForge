"""Small-data integration CLI. Explicit full-key joins, never row-order blending."""
import argparse
import json
from pathlib import Path
import polars as pl
from .metrics import ndcg, validate_rows

def reference(path):
    p = Path(path)
    frame = pl.read_parquet(str(p / '*.parquet') if p.is_dir() else p)
    if isinstance(frame.schema['item_id'], pl.List):
        fields = ['item_id'] + (['target'] if 'target' in frame.columns else [])
        frame = frame.select('feed_id', *fields).explode(fields)
    return frame

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['validate', 'evaluate']:
        p = sub.add_parser(name)
        p.add_argument('--predictions', required=True)
        p.add_argument('--reference', required=True)
    p = sub.add_parser('blend')
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--weights', nargs='+', type=float, required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'blend':
        import math
        if len(args.inputs) != len(args.weights) or any(not math.isfinite(w) or w < 0 for w in args.weights) or sum(args.weights) <= 0:
            raise ValueError('One finite nonnegative weight per input required')
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        models = [validate_rows(pl.read_parquet(p).iter_rows(named=True)) for p in args.inputs]
        if any(m.keys() != models[0].keys() for m in models):
            raise ValueError('Input key sets differ')
        total = sum(args.weights)
        rows = [{'feed_id': k[0], 'item_id': k[1], 'score': sum(w*m[k] for w,m in zip(args.weights, models))/total} for k in models[0]]
        frame = pl.DataFrame(rows).cast({'feed_id': pl.Int32, 'item_id': pl.Int32, 'score': pl.Float32})
        validate_rows(frame.iter_rows(named=True))
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(output)
        print(json.dumps({'rows': len(rows), 'output': str(output)}))
        return
    predicted = validate_rows(pl.read_parquet(args.predictions).iter_rows(named=True))
    ref = reference(args.reference)
    keys = list(ref.select('feed_id','item_id').iter_rows())
    if len(set(keys)) != len(keys) or set(keys) != predicted.keys():
        raise ValueError('Reference duplicates or prediction coverage mismatch')
    if args.command == 'validate':
        print(json.dumps({'valid': True, 'rows': len(keys)}))
        return
    groups = {}
    for row in ref.select('feed_id','item_id','target').iter_rows(named=True):
        ys, ss = groups.setdefault(row['feed_id'], ([], []))
        ys.append(row['target'])
        ss.append(predicted[(row['feed_id'], row['item_id'])])
    result = {'feeds': len(groups), 'rows': len(keys)}
    for name, cutoff in [('ndcg',None), ('ndcg_at_10',10), ('ndcg_at_30',30)]:
        result[name] = sum(ndcg(y,s,cutoff) for y,s in groups.values())/len(groups)
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
