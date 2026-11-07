"""B01: compare two checkpoint criteria on the same baseline training trajectory."""
import json
from pathlib import Path

import polars as pl
import torch
from .progress import ProgressLoader as DataLoader

from neuroforge.metrics import ndcg


def validation_scores(baseline, model, validation, config):
    loader = DataLoader(validation, batch_size=config['batch_size'], shuffle=False,
                        collate_fn=baseline.collate_feeds)
    scores, quality = [], []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            features, categories, users, padding, targets = batch
            device = config['device']
            logits = model(features.to(device), categories.to(device), users.to(device),
                           padding.to(device)).float().cpu()
            for i in range(len(users)):
                mask = ~padding[i]
                values = logits[i][mask].tolist()
                quality.append(ndcg(targets[i][mask].long().tolist(), values))
                scores.extend(values)
    if not quality or len(scores) != len(validation.keys):
        raise ValueError('Empty validation or key coverage mismatch')
    predictions = validation.keys.with_columns(pl.Series('score', scores, dtype=pl.Float32)).cast(
        {'feed_id': pl.Int32, 'item_id': pl.Int32})
    return sum(quality) / len(quality), predictions


class CheckpointSelection:
    """Wrap upstream epoch saving without changing optimizer, loss or architecture."""
    def __init__(self, baseline):
        self.baseline = baseline
        self.original_save = baseline.save_epoch
        self.best_ndcg = float('-inf')
        self.selected = {}

    def __call__(self, model, training, validation, output, config, metrics, history, best_loss):
        # DataLoader iteration draws a seed even without shuffling. Preserve the
        # training RNG so this extra measurement cannot change later batches.
        with torch.random.fork_rng(devices=[]):
            quality, predictions = validation_scores(self.baseline, model, validation, config)
        metrics = {**metrics, 'val_ndcg': quality}
        better_loss = metrics['val_logloss'] < best_loss
        # Original model.pt remains the Logloss-selected control checkpoint.
        best_loss = self.original_save(model, training, validation, output, config,
                                       metrics, history, best_loss)
        if better_loss:
            predictions.write_parquet(output / 'predictions_val_logloss.parquet')
            self.selected['logloss'] = dict(metrics)
        if quality > self.best_ndcg:
            self.best_ndcg = quality
            torch.save({'state_dict': model.state_dict(),
                        'categories_mapping': training.categories_mapping,
                        'epoch': metrics['epoch'],
                        'model_params': {k: config[k] for k in ('width', 'layers', 'dropout')}},
                       output / 'model_ndcg.pt')
            predictions.write_parquet(output / 'predictions_val_ndcg.parquet')
            self.selected['ndcg'] = dict(metrics)
        summary = {'selected': self.selected,
                   'selection_ndcg_difference': self.selected['ndcg']['val_ndcg'] - self.selected['logloss']['val_ndcg'],
                   'limitation': 'Selection-set comparison; not independent holdout evidence',
                   'best_val_ndcg': self.best_ndcg}
        (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(f"epoch={metrics['epoch']} val_ndcg={quality:.6f} "
              f"val_logloss={metrics['val_logloss']:.6f}", flush=True)
        return best_loss
