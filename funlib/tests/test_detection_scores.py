from funlib import evaluate
import numpy as np
import unittest


class TestDetectionScores(unittest.TestCase):

    def test_1d(self):

        truth = np.array([0, 1, 1, 0], dtype=np.uint64)
        test = np.array([0, 1, 2, 1], dtype=np.uint64)

        for matching_score, threshold in [
                ('overlap', 1),
                ('iou', 0.5),
                ('distance', 1.0)]:

            m = evaluate.detection_scores(
                truth,
                test,
                label_ids=[1, 2],
                matching_score=matching_score,
                matching_threshold=threshold)

            self.assertEqual(m['fp'], 2)
            self.assertEqual(m['fn'], 0)

            self.assertEqual(m['fp_1'], 1)
            self.assertEqual(m['fp_2'], 1)
            self.assertEqual(m['fn_1'], 0)
            self.assertEqual(m['fn_2'], 0)

    def test_2d(self):

        truth = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint64)
        test = np.array([[0, 1, 2, 1], [0, 1, 2, 1]], dtype=np.uint64)

        for matching_score, threshold in [
                ('overlap', 1),
                ('iou', 0.5),
                ('distance', 1.0)]:

            m = evaluate.detection_scores(
                truth,
                test,
                label_ids=[1, 2],
                matching_score=matching_score,
                matching_threshold=threshold,
                voxel_size=(10, 1))

            self.assertEqual(m['fp'], 2)
            self.assertEqual(m['fn'], 0)

            self.assertEqual(m['fp_1'], 1)
            self.assertEqual(m['fp_2'], 1)
            self.assertEqual(m['fn_1'], 0)
            self.assertEqual(m['fn_2'], 0)

    def test_return_matches(self):

        truth = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint64)
        test = np.array([[0, 1, 2, 1], [0, 1, 2, 1]], dtype=np.uint64)

        for matching_score, threshold in [
                ('overlap', 1),
                ('iou', 0.5),
                ('distance', 1.0)]:

            m = evaluate.detection_scores(
                truth,
                test,
                label_ids=[1, 2],
                matching_score=matching_score,
                matching_threshold=threshold,
                voxel_size=(10, 1),
                return_matches=True)

            self.assertEqual(m['fp'], 2)
            self.assertEqual(m['fn'], 0)

            self.assertEqual(m['fp_1'], 1)
            self.assertEqual(m['fp_2'], 1)
            self.assertEqual(m['fn_1'], 0)
            self.assertEqual(m['fn_2'], 0)

            self.assertEqual(len(m['matches_1']), 1)
            self.assertEqual(len(m['matches_2']), 0)
            self.assertEqual(len(np.unique(m['components_truth_1'])), 2)
            self.assertEqual(len(np.unique(m['components_truth_2'])), 1)
            self.assertEqual(len(np.unique(m['components_test_1'])), 3)
            self.assertEqual(len(np.unique(m['components_test_2'])), 2)

            self.assertTrue(
                np.equal(
                    m['components_truth_1'] == m['matches_1'][0][0],
                    truth == 1).all())
            self.assertTrue(
                np.equal(
                    m['components_test_1'] == m['matches_1'][0][1],
                    [
                        [False, True, False, False],
                        [False, True, False, False]
                    ]).all())
