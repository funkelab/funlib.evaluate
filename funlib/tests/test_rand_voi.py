from funlib import evaluate
import numpy as np
import unittest


class TestRandVoi(unittest.TestCase):

    def test_metric(self):

        a = np.array([1, 2, 3], dtype=np.uint64)
        b = np.array([4, 5, 6], dtype=np.uint64)

        m = evaluate.rand_voi(a, b)

        assert m['rand_split'] == 1.0
        assert m['rand_merge'] == 1.0
        assert m['voi_split'] == 0.0
        assert m['voi_merge'] == 0.0

        a = np.array([1, 1, 2, 2], dtype=np.uint64)
        b = np.array([2, 2, 2, 2], dtype=np.uint64)

        m = evaluate.rand_voi(a, b)

        assert m['rand_split'] == 1.0
        assert m['rand_merge'] == 0.5
        assert m['voi_split'] == 0.0
        assert m['voi_merge'] == 1.0

        m = evaluate.rand_voi(b, a)

        assert m['rand_split'] == 0.5
        assert m['rand_merge'] == 1.0
        assert m['voi_split'] == 1.0
        assert m['voi_merge'] == 0.0

        a = np.array([1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4], dtype=np.uint64)
        b = np.array([3, 3, 3, 3, 4, 5, 6, 6, 5, 5, 5], dtype=np.uint64)

        m = evaluate.rand_voi(a, b)

        self.assertAlmostEqual(m['rand_split'], 0.6363636363636364)
        self.assertAlmostEqual(m['rand_merge'], 0.5675675675675675)
        self.assertAlmostEqual(m['voi_split'], 0.6140806820148608)
        self.assertAlmostEqual(m['voi_merge'], 0.7272727272727271)

        m = evaluate.rand_voi(b, a, return_cluster_scores=True)

        self.assertAlmostEqual(m['rand_split'], 0.5675675675675675)
        self.assertAlmostEqual(m['rand_merge'], 0.6363636363636364)
        self.assertAlmostEqual(m['voi_split'], 0.7272727272727271)
        self.assertAlmostEqual(m['voi_merge'], 0.6140806820148608)

        self.assertAlmostEqual(sum(m['voi_split_i'].values()), m['voi_split'])
        self.assertAlmostEqual(sum(m['voi_merge_j'].values()), m['voi_merge'])

    def test_inputs(self):

        with self.assertRaises(AssertionError):
            a = np.array([1, 2, 3], dtype=np.uint64)
            b = np.array([[4, 5, 6]], dtype=np.uint64)
            evaluate.rand_voi(a, b)

        with self.assertRaises(AssertionError):
            a = np.array([1, 2, 3], dtype=np.uint64)
            b = np.array([4, 5], dtype=np.uint64)
            evaluate.rand_voi(a, b)

        with self.assertRaises(ValueError):
            a = np.array([1, 2, 3], dtype=np.uint32)
            b = np.array([4, 5, 6], dtype=np.uint64)
            evaluate.rand_voi(a, b)
