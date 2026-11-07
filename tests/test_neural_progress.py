import contextlib
import io
import unittest
import torch
from torch.utils.data import DataLoader
from neuroforge.models.neural.progress import ProgressLoader


class ProgressTests(unittest.TestCase):
    def test_progress_keeps_batch_order_and_rng(self):
        data = torch.arange(11)
        torch.manual_seed(42)
        expected = list(DataLoader(data, batch_size=2, shuffle=True))
        expected_rng = torch.random.get_rng_state()
        torch.manual_seed(42)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            actual = list(ProgressLoader(data, batch_size=2, shuffle=True))
        self.assertEqual([x.tolist() for x in actual], [x.tolist() for x in expected])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        self.assertIn('6/6', output.getvalue())
        self.assertIn('peak RAM=', output.getvalue())
