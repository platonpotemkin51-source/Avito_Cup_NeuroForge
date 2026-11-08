import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import polars as pl
import torch
from test_history_mean import fixture
from neuroforge.models.neural.history_mean import HistoryFeeds

ROOT=Path(__file__).resolve().parents[1]


class FullPipelineTests(unittest.TestCase):
    def test_control_can_skip_history_without_changing_other_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            normal=HistoryFeeds(root/'train')
            light=HistoryFeeds(root/'train',mapping=normal.mapping,include_history=False,release_pages=True)
            for a,b in zip(normal,light,strict=True):
                for k in ['features','category','user','target']:torch.testing.assert_close(a[k],b[k])
                self.assertTrue(torch.equal(b['history'],torch.zeros(129)))

    def test_synthetic_full_train_reload_test_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);data=root/'data';fixture(data)
            config=json.loads((ROOT/'configs/full_cpu_control.json').read_text())
            config.update(save_every_steps=1,threads=1)
            (root/'config.json').write_text(json.dumps(config))
            def run(script,*args):
                result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            output=root/'training'
            run('train_b02.py','--config',root/'config.json','--data',data,'--variant','control','--allow-large','--output',output)
            self.assertTrue((output/'model_latest.pt').exists())
            run('evaluate_b02.py','--data',data,'--model',output/'model.pt','--max-feeds','0','--threads','1','--output',root/'val')
            expected=pl.read_parquet(output/'predictions_val.parquet')
            self.assertTrue(expected.equals(pl.read_parquet(root/'val/predictions_val.parquet')))
            run('predict_b02.py','--data',data,'--model',output/'model.pt','--threads','1','--output',root/'test.parquet')
            prediction=pl.read_parquet(root/'test.parquet')
            self.assertEqual(prediction.height,5)
            self.assertTrue(prediction['score'].is_finite().all())
            self.assertTrue(prediction.equals(expected))
            self.assertFalse(any(p.name.startswith('predict-parts-') for p in root.iterdir()))
