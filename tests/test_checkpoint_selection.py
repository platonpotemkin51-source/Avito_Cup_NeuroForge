import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import polars as pl
import torch
from neuroforge.models.neural.checkpoint_selection import CheckpointSelection

ROOT = Path(__file__).resolve().parents[1]


class CheckpointTests(unittest.TestCase):
    def test_different_epochs_selected_without_changing_training(self):
        spec = importlib.util.spec_from_file_location('original_nn_test', ROOT / 'baseline/neural_baseline.py')
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        model = baseline.Ranker(1)
        data = SimpleNamespace(categories_mapping={1: 1}, keys=[(1, 1), (1, 2)])
        config = {'width': 64, 'layers': 2, 'dropout': .1}
        history = []
        predictions = pl.DataFrame({'feed_id': [1, 1], 'item_id': [1, 2], 'score': [1., 0.]})
        selector = CheckpointSelection(baseline)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('neuroforge.models.neural.checkpoint_selection.validation_scores',
                       side_effect=[(.8, predictions), (.7, predictions), (.7, predictions)]):
                best = float('inf')
                for epoch, loss in enumerate([.5, .4, .4], 1):
                    best = selector(model, data, data, output, config,
                                    {'epoch': epoch, 'val_logloss': loss}, history, best)
            self.assertEqual(torch.load(output / 'model.pt', weights_only=True)['epoch'], 2)
            self.assertEqual(torch.load(output / 'model_ndcg.pt', weights_only=True)['epoch'], 1)
            summary = json.loads((output / 'metrics.json').read_text())
            self.assertAlmostEqual(summary['selection_ndcg_difference'], .1)
            self.assertEqual(len(history), 3)
            self.assertEqual(best, .4)
            self.assertTrue((output / 'predictions_val_ndcg.parquet').exists())
