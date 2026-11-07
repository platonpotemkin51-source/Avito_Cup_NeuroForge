import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

import polars as pl
import torch

spec = importlib.util.spec_from_file_location('evaluate_neural', Path(__file__).resolve().parents[1] / 'scripts/evaluate_neural.py')
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class EvaluationTests(unittest.TestCase):
    def test_padding_keys_and_equal_feed_weight(self):
        class Dataset:
            keys = pl.DataFrame({'feed_id': [9, 9, 4, 4, 4], 'item_id': [8, 2, 7, 3, 6]})

        class Model(torch.nn.Module):
            def forward(self, features, categories, users, padding):
                return torch.tensor([[0., 1., 999.], [3., 2., 1.]])

        batch = (torch.zeros(2, 3, 134), torch.zeros(2, 3, dtype=torch.long),
                 torch.zeros(2, 128), torch.tensor([[False, False, True], [False, False, False]]),
                 torch.tensor([[1., 0., 1.], [1., 0., 0.]]))
        predictions, metrics = evaluator.evaluate(Model(), Dataset(), [batch])
        self.assertEqual(predictions['item_id'].to_list(), [8, 2, 7, 3, 6])
        self.assertEqual(predictions['score'].to_list(), [0., 1., 3., 2., 1.])
        self.assertAlmostEqual(metrics['ndcg'], (1 / math.log2(3) + 1) / 2)
        self.assertEqual(metrics['feeds'], 2)
        self.assertEqual(metrics['candidates'], 5)
        self.assertEqual(predictions.schema['score'], pl.Float32)

    def test_existing_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['evaluate', '--data', 'unused', '--model', 'unused', '--output', directory]):
                with self.assertRaises(FileExistsError):
                    evaluator.main()
