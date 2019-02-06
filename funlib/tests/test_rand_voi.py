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

        a = np.array([1, 1, 2, 2, 2], dtype=np.uint64)
        b = np.array([3, 3, 3, 4, 4], dtype=np.uint64)

        m = evaluate.rand_voi(a, b)

        self.assertAlmostEqual(m['rand_split'], 0.6923076923076923)
        self.assertAlmostEqual(m['rand_merge'], 0.6923076923076923)
        self.assertAlmostEqual(m['voi_split'], 0.5509775004326936)
        self.assertAlmostEqual(m['voi_merge'], 0.5509775004326936)

        m = evaluate.rand_voi(b, a)

        self.assertAlmostEqual(m['rand_split'], 0.6923076923076923)
        self.assertAlmostEqual(m['rand_merge'], 0.6923076923076923)
        self.assertAlmostEqual(m['voi_split'], 0.5509775004326936)
        self.assertAlmostEqual(m['voi_merge'], 0.5509775004326936)

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
