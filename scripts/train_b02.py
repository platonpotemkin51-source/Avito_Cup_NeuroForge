"""CPU B02 runner. No full train is launched unless --allow-large is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import polars as pl
import torch
from torch.nn import functional as F
from neuroforge.metrics import ndcg
from neuroforge.models.neural.history_mean import HistoryFeeds, collate_history, make_ranker
from neuroforge.models.neural.progress import ProgressLoader, peak_ram_gib


def forward(model, batch):
    return model(batch['features'], batch['category'], batch['users'], batch['padding'], batch['history'])


def evaluate(model, dataset, batch_size):
    model.eval()
    frames, totals = [], {'ndcg': 0., 'ndcg_at_10': 0., 'ndcg_at_30': 0.}
    feeds, count, loss = 0, 0, 0.
    loader = ProgressLoader(dataset, batch_size=batch_size, collate_fn=collate_history, num_workers=0)
    with torch.inference_mode():
        for batch in loader:
            logits = forward(model, batch).float()
            real = ~batch['padding']
            loss += F.binary_cross_entropy_with_logits(logits[real], batch['target'][real], reduction='sum').item()
            values = logits[real].tolist()
            frame = pl.DataFrame({'feed_id': [k[0] for k in batch['keys']],
                                  'item_id': [k[1] for k in batch['keys']], 'score': values},
                                 schema={'feed_id': pl.Int32, 'item_id': pl.Int32, 'score': pl.Float32})
            frames.append(frame)
            count += len(values)
            for i in range(len(logits)):
                mask = real[i]
                yy, ss = batch['target'][i][mask].long().tolist(), logits[i][mask].tolist()
                for name, cutoff in [('ndcg', None), ('ndcg_at_10', 10), ('ndcg_at_30', 30)]:
                    totals[name] += ndcg(yy, ss, cutoff)
                feeds += 1
    if feeds != len(dataset):
        raise ValueError('Evaluation feed coverage mismatch')
    return {**{k: v / feeds for k, v in totals.items()}, 'logloss': loss / count,
            'feeds': feeds, 'candidates': count}, pl.concat(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/B02_cpu_smoke.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['control', 'history'], required=True)
    parser.add_argument('--allow-large', action='store_true', help='Explicitly enable over 2000 training feeds or full train')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    if config['train_feeds'] < 0 or config['val_feeds'] < 0 or min(config['epochs'], config['batch_size'], config['threads']) < 1:
        raise ValueError('Invalid size or count')
    if (config['train_feeds'] == 0 or config['train_feeds'] > 2000) and not args.allow_large:
        parser.error('Large training is disabled; review profiling before using --allow-large')
    started = time.perf_counter()
    torch.set_num_threads(config['threads'])
    torch.manual_seed(config['seed'])
    args.output.mkdir(parents=True)
    resolved = {**config, 'variant': args.variant, 'device': 'cpu', 'data': str(args.data.resolve())}
    (args.output / 'config.json').write_text(json.dumps(resolved, indent=2)+'\n')
    print('[loading] Building train inventory/category vocabulary', flush=True)
    input_options = {'include_history': args.variant == 'history' or not config.get('skip_unused_history', False),
                     'release_pages': config.get('release_pages', False)}
    training = HistoryFeeds(args.data / 'train', config['train_feeds'], shuffle=True, seed=config['seed'], **input_options)
    print('[loading] Building validation inventory', flush=True)
    validation = HistoryFeeds(args.data / 'val', config['val_feeds'], mapping=training.mapping, **input_options)
    print(f'[loaded] train={len(training)} val={len(validation)}; RAM={peak_ram_gib():.2f} GiB', flush=True)
    model_params = {k: config[k] for k in ['width', 'layers', 'dropout']}
    model = make_ranker(len(training.mapping), use_history=args.variant == 'history', **model_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    history, best = [], float('-inf')
    def save_weights(name, epoch, step=None):
        temporary = args.output / (name + '.tmp')
        torch.save({'state_dict': model.state_dict(), 'categories_mapping': training.mapping,
                    'model_params': model_params, 'use_history': args.variant == 'history',
                    'epoch': epoch, 'step': step, 'completed_epoch': step is None}, temporary)
        temporary.replace(args.output / name)

    for epoch in range(1, config['epochs'] + 1):
        training.epoch = epoch - 1
        model.train()
        total_loss, candidates = 0., 0
        loader = ProgressLoader(training, batch_size=config['batch_size'], collate_fn=collate_history, num_workers=0)
        loader.stage = 'train'
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            real = ~batch['padding']
            logits = forward(model, batch)
            loss = F.binary_cross_entropy_with_logits(logits[real], batch['target'][real])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            n = int(real.sum())
            total_loss += loss.item() * n
            candidates += n
            if config.get('save_every_steps', 0) and step % config['save_every_steps'] == 0:
                save_weights('model_latest.pt', epoch, step)
                print(f'[saved] intermediate inference weights: epoch={epoch}, batch={step}', flush=True)
        save_weights('model_latest.pt', epoch)
        with torch.random.fork_rng(devices=[]):
            quality, predictions = evaluate(model, validation, config['batch_size'])
        metrics = {'epoch': epoch, 'train_logloss': total_loss / candidates, **quality}
        history.append(metrics)
        if quality['ndcg'] > best:
            best = quality['ndcg']
            torch.save({'state_dict': model.state_dict(), 'categories_mapping': training.mapping,
                        'model_params': model_params, 'use_history': args.variant == 'history',
                        'epoch': epoch}, args.output / 'model.pt')
            predictions.write_parquet(args.output / 'predictions_val.parquet')
            (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        (args.output / 'history.json').write_text(json.dumps(history, indent=2)+'\n')
        print(json.dumps(metrics), flush=True)
    manifest = {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'dirty_state': subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
                'python': platform.python_version(), 'torch': torch.__version__,
                'parameters': sum(p.numel() for p in model.parameters()), 'seconds': time.perf_counter()-started,
                'peak_ram_gib': peak_ram_gib(), 'device': 'cpu', 'train_feeds': len(training),
                'val_feeds': len(validation), 'selection': 'maximum full NDCG on supplied val selection',
                'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__).resolve(),ROOT/'src/neuroforge/models/neural/history_mean.py',
                                            ROOT/'baseline/neural_baseline.py',ROOT/'src/neuroforge/metrics.py',
                                            ROOT/'src/neuroforge/models/neural/progress.py']},
                'parts': {name:[{'name':p.name,'feeds':n} for p,n in ds.parts]
                          for name,ds in [('train',training),('val',validation)]}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output / 'report.md').write_text('# B02 '+args.variant+'\n\nSee config.json, metrics.json, history.json and manifest.json.\nSmall bounded runs verify execution only; no claim of ranking improvement.\n')
    print(f'[finished] seconds={manifest["seconds"]:.2f} peak RAM={peak_ram_gib():.2f} GiB',flush=True)


if __name__ == '__main__':
    main()
