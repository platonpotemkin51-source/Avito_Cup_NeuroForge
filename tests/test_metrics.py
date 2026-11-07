import math
import unittest
from neuroforge.metrics import ndcg, validate_rows

class MetricTests(unittest.TestCase):
    def test_ideal(self):
        self.assertEqual(ndcg([1,0,1], [3,1,2]), 1)
    def test_pessimistic_tie(self):
        self.assertAlmostEqual(ndcg([1,0], [1,1]), 1/math.log2(3))
    def test_no_positive(self):
        self.assertEqual(ndcg([0,0], [1,2]),1)
    def test_cutoff(self):
        self.assertEqual(ndcg([1,0], [1,2],1),0)
    def test_multiple_positive(self):
        self.assertAlmostEqual(ndcg([1,0,1], [3,2,1]), 1.5/(1+1/math.log2(3)))
    def test_invalid(self):
        for y,s in [([],[]),([1],[float('nan')]),([2],[1]),([1],[1,2])]:
            with self.assertRaises(ValueError): ndcg(y,s)
    def test_duplicate(self):
        row = {'feed_id':1,'item_id':2,'score':0.5}
        with self.assertRaises(ValueError): validate_rows([row,row])
    def test_nonfinite(self):
        with self.assertRaises(ValueError): validate_rows([{'feed_id':1,'item_id':2,'score':float('inf')}])

if __name__ == '__main__': unittest.main()
