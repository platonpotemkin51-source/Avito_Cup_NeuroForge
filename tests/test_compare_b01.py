import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

spec = importlib.util.spec_from_file_location('compare_b01', Path(__file__).resolve().parents[1] / 'scripts/compare_b01.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_selection_excluded_and_predictions_joined_by_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data/val/impressions'
            data.mkdir(parents=True)
            pl.DataFrame({'feed_id': [1, 2], 'item_id': [[10,11], [20,21]],
                          'target': [[1,0], [0,1]], 'user_id': [8,9],
                          'timestamp': [100,100]}).cast({'feed_id': pl.Int32}).write_parquet(data / 'part-0.parquet')
            run = root / 'run'
            run.mkdir()
            pl.DataFrame({'feed_id': [1]}).write_parquet(run / 'predictions_val_ndcg.parquet')
            (run / 'metrics.json').write_text(json.dumps({'selected': {}}))
            for criterion, scores in [('logloss', [1.,0.,1.,0.]), ('ndcg', [0.,1.,0.,1.])]:
                p = run / f'val-full-{criterion}'
                p.mkdir()
                pl.DataFrame({'feed_id': [1,1,2,2], 'item_id': [10,11,20,21], 'score': scores}).cast(
                    {'feed_id': pl.Int32}).reverse().write_parquet(p / 'predictions.parquet')
            output = root / 'comparison'
            with patch.object(sys, 'argv', ['compare', '--data', str(root/'data'), '--runs', str(run), '--output', str(output)]), contextlib.redirect_stdout(io.StringIO()):
                comparison.main()
            result = json.loads((output / 'comparison.json').read_text())['run']
            self.assertEqual(result['remaining_val']['feeds'], 1)
            self.assertEqual(result['remaining_val']['ndcg_selected_ndcg'], 1.)
            self.assertLess(result['selection_100']['difference'], 0)
            self.assertGreater(result['remaining_val']['difference'], 0)
            self.assertAlmostEqual(result['full_val']['difference'], 0.)
            self.assertGreater(result['full_val']['random_expected_ndcg'], 0)
