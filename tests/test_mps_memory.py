import copy
import unittest
from unittest.mock import patch
import torch
from neuroforge.models.neural.history_mean import collate_history
from neuroforge.models.neural.signal_ranker import SignalRanker
from neuroforge.models.neural.mps_memory import housekeeping


class MemoryTests(unittest.TestCase):
    def test_fixed_padding_masks_and_real_logits(self):
        torch.set_num_threads(1);torch.manual_seed(40)
        rows=[]
        for i,n in enumerate([3,1]):
            rows.append({'features':torch.randn(n,134),'category':torch.zeros(n,dtype=torch.long),
                         'user':torch.randn(128),'history':torch.zeros(129),'signals':torch.rand(n,3),
                         'item_id':list(range(n)),'feed_id':i,'target':torch.zeros(n)})
        plain=collate_history(rows)
        padded_rows=copy.deepcopy(rows)
        for row in padded_rows:row['pad_to']=6
        fixed=collate_history(padded_rows)
        self.assertEqual(fixed['features'].shape,(2,6,134))
        self.assertEqual(fixed['padding'].sum().item(),8)
        self.assertTrue((fixed['signals'][fixed['padding']]==0).all())
        model=SignalRanker(1,width=8,layers=1,dropout=0).eval()
        def run(batch):return model(batch['features'],batch['category'],batch['users'],batch['padding'],batch['history'],batch['signals'])
        with torch.inference_mode():
            a=run(plain)[~plain['padding']];b=run(fixed)[~fixed['padding']]
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        for row in padded_rows:row['pad_to']=1
        with self.assertRaisesRegex(ValueError,'exceeds'):collate_history(padded_rows)

    def test_cache_cleanup_and_budget_guard(self):
        self.assertIsNone(housekeeping('cpu','test'))
        with patch('torch.mps.synchronize'),patch('torch.mps.empty_cache') as clear,patch('torch.mps.driver_allocated_memory',side_effect=[12*1024**3,2*1024**3]),patch('torch.mps.current_allocated_memory',return_value=1024**3):
            result=housekeeping('mps','test');clear.assert_called_once();self.assertEqual(result['driver_after_gib'],2.)
        with patch('torch.mps.synchronize'),patch('torch.mps.empty_cache'),patch('torch.mps.driver_allocated_memory',return_value=9*1024**3),patch('torch.mps.current_allocated_memory',return_value=1024**3):
            with self.assertRaisesRegex(RuntimeError,'budget'):housekeeping('mps','test',8.)
