import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import polars as pl
import torch
from test_history_mean import fixture
from neuroforge.models.neural.history_mean import HistoryFeeds,collate_history
from neuroforge.models.neural.data_signals import extract_signals
from neuroforge.models.neural.signal_ranker import SignalRanker


class RetainedTests(unittest.TestCase):
    def test_top3_missing_order_and_recent_window(self):
        table=np.eye(128,dtype=np.int8)*100;items=np.eye(128,dtype=np.float32)[:3];user=items[0]
        values=extract_signals(table,[0,1,2,-1],items,user)
        np.testing.assert_allclose(values[:,2],np.ones(3)/3)
        np.testing.assert_array_equal(extract_signals(table,[],items,user)[:,1:],0)
        np.testing.assert_array_equal(extract_signals(table,[-1]*32+[0],items,user)[:,1:],0)
        perm=[2,0,1];np.testing.assert_array_equal(extract_signals(table,[0,1,2],items[perm],user),values[perm])

    def test_streaming_label_independence_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            a=list(HistoryFeeds(root/'train',with_signals=True,pad_to=6))
            for path in (root/'train/impressions').glob('*.parquet'):
                frame=pl.read_parquet(path).with_columns(pl.col('target').list.eval(1-pl.element()));frame.write_parquet(path)
            b=list(HistoryFeeds(root/'train',with_signals=True,pad_to=6,with_labels=False))
            for x,y in zip(a,b):
                torch.testing.assert_close(x['signals'],y['signals'],rtol=0,atol=0);self.assertIsNone(y['target'])
            batch=collate_history(b);self.assertEqual(batch['signals'].shape,(2,6,3));self.assertIsNone(batch['target'])
            self.assertTrue((batch['signals'][batch['padding']]==0).all())

    def test_checkpoint_capacity_and_gradients(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        model=SignalRanker(2,width=8,layers=1,dropout=0)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            rows=list(HistoryFeeds(root/'train',with_signals=True,pad_to=6));b=collate_history(rows)
            def run(m,x):return m(x['features'],x['category'],x['users'],x['padding'],x['history'],x['signals'])
            model.eval();scores=run(model,b)
            changed=copy.deepcopy(b);changed['features'][b['padding']]=999;changed['signals'][b['padding']]=999
            torch.testing.assert_close(run(model,changed)[~b['padding']],scores[~b['padding']])
            mask=~b['padding'];loss=torch.nn.functional.binary_cross_entropy_with_logits(scores[mask],b['target'][mask]);loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            restored=SignalRanker(2,width=8,layers=1,dropout=0);restored.load_state_dict(model.state_dict());restored.eval()
            torch.testing.assert_close(run(restored,b),scores)
            with self.assertRaises(ValueError):SignalRanker(2,ranks='history')
