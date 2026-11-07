import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import polars as pl
from neuroforge.cli import main


class CliTests(unittest.TestCase):
    def call(self, *args):
        out = io.StringIO()
        with patch.object(sys, 'argv', ['cli', *map(str, args)]), contextlib.redirect_stdout(out):
            main()
        return json.loads(out.getvalue())

    def test_full_pipeline_and_key_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            pl.DataFrame({'feed_id': [1,2], 'item_id': [[10,11],[20,21]],
                          'target': [[1,0],[0,1]]}).write_parquet(p/'labels.parquet')
            a = pl.DataFrame({'feed_id':[1,1,2,2], 'item_id':[10,11,20,21], 'score':[3.,0.,0.,3.]})
            a.write_parquet(p/'a.parquet')
            a.reverse().write_parquet(p/'b.parquet')
            self.call('blend','--inputs',p/'a.parquet',p/'b.parquet','--weights',1,1,'--output',p/'blend.parquet')
            result = self.call('evaluate','--predictions',p/'blend.parquet','--reference',p/'labels.parquet')
            self.assertEqual(result['ndcg'],1.)
            self.assertEqual(result['feeds'],2)
            self.assertTrue(self.call('validate','--predictions',p/'blend.parquet','--reference',p/'a.parquet')['valid'])
            a.head(3).write_parquet(p/'bad.parquet')
            with self.assertRaises(ValueError):
                self.call('evaluate','--predictions',p/'bad.parquet','--reference',p/'labels.parquet')
            with self.assertRaises(FileExistsError):
                self.call('blend','--inputs',p/'a.parquet','--weights',1,'--output',p/'blend.parquet')

if __name__ == '__main__':
    unittest.main()
