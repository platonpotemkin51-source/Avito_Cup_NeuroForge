import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import polars as pl
from neuroforge.metrics import ndcg

spec=importlib.util.spec_from_file_location('compare_b02_test',Path(__file__).resolve().parents[1]/'scripts/compare_b02.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


def fixture(root):
    data=root/'data/val/impressions';data.mkdir(parents=True)
    pl.DataFrame({'feed_id':[1,2],'user_id':[3,4],'item_id':[[10,11],[20,21]],
                  'target':[[1,0],[0,1]],'history__item_id':[[],[7]]}).cast({'feed_id':pl.Int32}).write_parquet(data/'part.parquet')
    for variant,scores in [('control',[1.,0.,1.,0.]),('history',[0.,1.,0.,1.])]:
        p=root/variant;p.mkdir()
        cfg={'variant':variant,'epochs':1,'seed':42}
        metric={'epoch':1,'ndcg':(ndcg([1,0],scores[:2])+ndcg([0,1],scores[2:]))/2}
        (p/'config.json').write_text(json.dumps(cfg))
        (p/'metrics.json').write_text(json.dumps(metric))
        (p/'history.json').write_text(json.dumps([metric]))
        manifest={'seconds':1.,'peak_ram_gib':.1,'parameters':10,'parts':{},'source_sha256':{},'python':'3.11','torch':'2.10'}
        (p/'manifest.json').write_text(json.dumps(manifest))
        pl.DataFrame({'feed_id':[1,1,2,2],'item_id':[10,11,20,21],'score':scores}).cast({'feed_id':pl.Int32,'score':pl.Float32}).reverse().write_parquet(p/'predictions_val.parquet')


class PairAuditTests(unittest.TestCase):
    def test_pair_audit_keys_and_result(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                result=comparison.compare(root/'data',root/'control',root/'history',root/'out')
            self.assertAlmostEqual(result['difference'],0.)
            self.assertEqual(result['feeds'],2)
            self.assertTrue((root/'out/per_feed_comparison.parquet').exists())

    def test_rejects_different_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            p=root/'history/config.json';c=json.loads(p.read_text());c['seed']=123;p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'Conditions'):
                comparison.compare(root/'data',root/'control',root/'history',root/'out')

    def test_rejects_incomplete_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root)
            p=root/'control/config.json';c=json.loads(p.read_text());c['epochs']=2;p.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                comparison.compare(root/'data',root/'control',root/'history',root/'out')
