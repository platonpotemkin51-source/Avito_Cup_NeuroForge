import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from neuroforge.models.neural.history_mean import HistoryFeeds, mean_history, vectors, collate_history, make_ranker


def fixture(root):
    for split in ['train', 'val', 'test']:
        p = root / split
        (p / 'impressions').mkdir(parents=True)
        (p / 'item_embeddings').mkdir()
        table = np.zeros((3,128), dtype=np.int8)
        table[0,0] = 127
        table[1,1] = 127
        table.tofile(p / 'item_embeddings/data.memmap')
        (p / 'item_embeddings/data.meta').write_text(json.dumps({'dtype':'int8','shape':[3,128],'order':'C'}))
        for part in range(2):
            n = part + 2
            row = {'feed_id':part+1,'item_id':list(range(10,10+n)), 'category_id':[part+1]*n,
                   'item_id__embedding_idx':[0,-1]+([1] if part else []), 'user_embedding':[1]*128,
                   'history__item_id__embedding_idx':[0,1,-1] if part == 0 else []}
            row.update({f'f{k}':[.1]*n for k in range(1,6)})
            if split != 'test':
                row['target']=[1]+[0]*(n-1)
            pl.DataFrame([row]).write_parquet(p / f'impressions/part-{part}.parquet')


class HistoryTests(unittest.TestCase):
    def test_average_missing_empty_zero_and_int8_safety(self):
        table=np.zeros((3,128),dtype=np.int8)
        table[0,0]=127; table[1,1]=127
        expected=torch.zeros(129);expected[:2]=.5
        torch.testing.assert_close(mean_history(table,[0,1,-1,2]),expected)
        torch.testing.assert_close(mean_history(table,[2,-1]),mean_history(table,[]))
        self.assertEqual(mean_history(table,[])[128].item(),1.)
        self.assertEqual(mean_history(table[:2],[-1])[128].item(),1.)
        torch.testing.assert_close(mean_history(table,[0,1]),mean_history(table,[1,0]))
        torch.testing.assert_close(mean_history(table,[0,1]),mean_history(table,[0,1,0,1]))
        with self.assertRaises(ValueError): vectors(table,[-2])
        with self.assertRaises(ValueError): vectors(table,[3])

    def test_shards_limits_unknowns_and_unlabelled_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root)
            train=HistoryFeeds(root/'train',1,shuffle=True)
            self.assertEqual(len(list(train)),1)
            self.assertEqual(train.mapping,{1:1})
            val=HistoryFeeds(root/'val',mapping=train.mapping)
            rows=list(val)
            self.assertEqual([r['feed_id'] for r in rows],[1,2])
            self.assertEqual(rows[1]['category'].tolist(),[0,0,0])
            self.assertEqual(rows[0]['features'][1,-1].item(),1.)
            self.assertEqual(collate_history(rows)['padding'].tolist(),[[False,False,True],[False,False,False]])
            self.assertIsNone(collate_history(list(HistoryFeeds(root/'test',mapping=train.mapping)))['target'])
            shuffled=HistoryFeeds(root/'train',shuffle=True,seed=123)
            self.assertEqual(sorted(r['feed_id'] for r in shuffled),[1,2])
            self.assertEqual([r['feed_id'] for r in shuffled],[r['feed_id'] for r in shuffled])

    def test_matched_capacity_permutation_padding_gradients_reload(self):
        torch.set_num_threads(1)
        torch.manual_seed(42); control=make_ranker(2,use_history=False)
        torch.manual_seed(42); model=make_ranker(2,use_history=True)
        self.assertEqual(sum(p.numel() for p in control.parameters()),sum(p.numel() for p in model.parameters()))
        for k,v in model.state_dict().items(): torch.testing.assert_close(v,control.state_dict()[k])
        model.eval(); control.eval()
        x=torch.randn(1,4,134);c=torch.ones(1,4,dtype=torch.long);u=torch.randn(1,128)
        pad=torch.tensor([[False,False,False,True]]); h=torch.randn(1,129)
        y=model(x,c,u,pad,h)
        torch.testing.assert_close(control(x,c,u,pad,h),model(x,c,u,pad,torch.zeros_like(h)))
        torch.testing.assert_close(control(x,c,u,pad,h),control(x,c,u,pad,h*20))
        self.assertFalse(torch.allclose(y[:,:3],model(x,c,u,pad,torch.zeros_like(h))[:,:3]))
        perm=torch.tensor([2,0,3,1])
        torch.testing.assert_close(model(x[:,perm],c[:,perm],u,pad[:,perm],h),y[:,perm],atol=1e-6,rtol=1e-5)
        changed=x.clone();changed[:,3]=999
        torch.testing.assert_close(model(changed,c,u,pad,h)[:,:3],y[:,:3])
        loss=F.binary_cross_entropy_with_logits(y[~pad],torch.tensor([1.,0.,0.]))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        restored=make_ranker(2,use_history=True);restored.load_state_dict(model.state_dict());restored.eval()
        torch.testing.assert_close(restored(x,c,u,pad,h),y)
