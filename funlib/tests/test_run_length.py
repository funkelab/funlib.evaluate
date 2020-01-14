from funlib import evaluate
import math
import networkx
import unittest


class TestRandVoi(unittest.TestCase):

    def test_skeleton_lengths(self):

        skeletons = networkx.Graph()

        # length of 2*sqrt(2)
        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)

        # length of 1*sqrt(2)
        skeletons.add_node(4, skeleton_id=2, z=1, y=1, x=1)
        skeletons.add_node(5, skeleton_id=2, z=0, y=0, x=1)
        skeletons.add_edge(4, 5)

        skeleton_lengths = evaluate.get_skeleton_lengths(
            skeletons,
            skeleton_position_attributes=['z', 'y', 'x'],
            skeleton_id_attribute='skeleton_id')

        self.assertAlmostEqual(skeleton_lengths[1], 2*math.sqrt(2))
        self.assertAlmostEqual(skeleton_lengths[2], math.sqrt(2))

    def test_edge_scores(self):

        skeletons = networkx.Graph()

        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)

        skeletons.add_node(4, skeleton_id=2, z=1, y=1, x=1)
        skeletons.add_node(5, skeleton_id=2, z=0, y=0, x=1)
        skeletons.add_edge(4, 5)

        # perfect match
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
            4: 20,
            5: 20
        }

        scores = evaluate.evaluate_skeletons(
            skeletons=skeletons,
            skeleton_id_attribute='skeleton_id',
            node_segment_lut=node_segment_lut)

        self.assertEqual(scores[1].ommitted, 0)
        self.assertEqual(scores[1].split, 0)
        self.assertEqual(scores[1].merged, 0)
        self.assertEqual(scores[1].correct, 2)

        self.assertEqual(scores[2].ommitted, 0)
        self.assertEqual(scores[2].split, 0)
        self.assertEqual(scores[2].merged, 0)
        self.assertEqual(scores[2].correct, 1)

        # one split in 1
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 30,
            4: 20,
            5: 20
        }

        scores = evaluate.evaluate_skeletons(
            skeletons=skeletons,
            skeleton_id_attribute='skeleton_id',
            node_segment_lut=node_segment_lut)

        self.assertEqual(scores[1].ommitted, 0)
        self.assertEqual(scores[1].split, 1)
        self.assertEqual(scores[1].merged, 0)
        self.assertEqual(scores[1].correct, 1)

        self.assertEqual(scores[2].ommitted, 0)
        self.assertEqual(scores[2].split, 0)
        self.assertEqual(scores[2].merged, 0)
        self.assertEqual(scores[2].correct, 1)

        # one split in 1, one in 2
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 30,
            4: 20,
            5: 40
        }

        scores = evaluate.evaluate_skeletons(
            skeletons=skeletons,
            skeleton_id_attribute='skeleton_id',
            node_segment_lut=node_segment_lut)

        self.assertEqual(scores[1].ommitted, 0)
        self.assertEqual(scores[1].split, 1)
        self.assertEqual(scores[1].merged, 0)
        self.assertEqual(scores[1].correct, 1)

        self.assertEqual(scores[2].ommitted, 0)
        self.assertEqual(scores[2].split, 1)
        self.assertEqual(scores[2].merged, 0)
        self.assertEqual(scores[2].correct, 0)

        # two splits in 1, one in 2
        node_segment_lut = {
            1: 10,
            2: 50,
            3: 30,
            4: 20,
            5: 40
        }

        scores = evaluate.evaluate_skeletons(
            skeletons=skeletons,
            skeleton_id_attribute='skeleton_id',
            node_segment_lut=node_segment_lut)

        self.assertEqual(scores[1].ommitted, 0)
        self.assertEqual(scores[1].split, 2)
        self.assertEqual(scores[1].merged, 0)
        self.assertEqual(scores[1].correct, 0)

        self.assertEqual(scores[2].ommitted, 0)
        self.assertEqual(scores[2].split, 1)
        self.assertEqual(scores[2].merged, 0)
        self.assertEqual(scores[2].correct, 0)

        # complete merge
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 10
        }

        scores = evaluate.evaluate_skeletons(
            skeletons=skeletons,
            skeleton_id_attribute='skeleton_id',
            node_segment_lut=node_segment_lut)

        self.assertEqual(scores[1].ommitted, 0)
        self.assertEqual(scores[1].split, 0)
        self.assertEqual(scores[1].merged, 2)
        self.assertEqual(scores[1].correct, 0)

        self.assertEqual(scores[2].ommitted, 0)
        self.assertEqual(scores[2].split, 0)
        self.assertEqual(scores[2].merged, 1)
        self.assertEqual(scores[2].correct, 0)

    def test_expected_run_length(self):

        skeletons = networkx.Graph()

        # one skeleton, 2*sqrt(2), no errors

        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
        }

        erl = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'])

        for u, v, data in skeletons.edges(data=True):
            self.assertAlmostEqual(data['length'], math.sqrt(2), places=5)

        self.assertAlmostEqual(erl, 2*math.sqrt(2), places=5)

        # ...same, one split error

        node_segment_lut = {
            1: 10,
            2: 10,
            3: 20,
        }

        erl = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'])

        for u, v, data in skeletons.edges(data=True):
            self.assertAlmostEqual(data['length'], math.sqrt(2), places=5)

        self.assertAlmostEqual(erl, 0.5*math.sqrt(2), places=5)

        # one split in 1, one in 2

        skeletons = networkx.Graph()
        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)
        skeletons.add_node(4, skeleton_id=2, z=1, y=1, x=1)
        skeletons.add_node(5, skeleton_id=2, z=0, y=0, x=1)
        skeletons.add_edge(4, 5)
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 30,
            4: 20,
            5: 40
        }

        erl = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'])

        self.assertAlmostEqual(erl, 2.0/3*0.5*math.sqrt(2), places=5)

        # complete merge
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 10
        }

        erl = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'])

        self.assertAlmostEqual(erl, 0)

        # skeleton: o--o--o--o--o (distance = 1 between nodes)
        #
        # labels A: 1  1  1  2  2 -> ERL = 1.25
        # labels B: 1  1  1  1  2 -> ERL = 2.25

        skeletons = networkx.Graph()
        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=0, y=0, x=1)
        skeletons.add_node(3, skeleton_id=1, z=0, y=0, x=2)
        skeletons.add_node(4, skeleton_id=1, z=0, y=0, x=3)
        skeletons.add_node(5, skeleton_id=1, z=0, y=0, x=4)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)
        skeletons.add_edge(3, 4)
        skeletons.add_edge(4, 5)
        node_segment_lut_A = {
            1: 1,
            2: 1,
            3: 1,
            4: 2,
            5: 2
        }
        node_segment_lut_B = {
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 2
        }

        erl_A = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut_A,
            skeleton_position_attributes=['z', 'y', 'x'])
        erl_B = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut_B,
            skeleton_position_attributes=['z', 'y', 'x'])

        self.assertAlmostEqual(erl_A, 1.25, places=5)
        self.assertAlmostEqual(erl_B, 2.25, places=5)

    def test_expected_run_length_precomputed_lengths(self):

        skeletons = networkx.Graph()

        # one skeleton, 2*sqrt(2), no errors

        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
        }

        skeleton_lengths = evaluate.get_skeleton_lengths(
            skeletons,
            ['z', 'y', 'x'],
            'skeleton_id',
            store_edge_length='length')

        erl = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_lengths=skeleton_lengths)

        for u, v, data in skeletons.edges(data=True):
            self.assertAlmostEqual(data['length'], math.sqrt(2), places=5)

        self.assertAlmostEqual(erl, 2*math.sqrt(2), places=5)

        # ...same, one split error

        node_segment_lut = {
            1: 10,
            2: 10,
            3: 20,
        }

        erl = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_lengths=skeleton_lengths)

        for u, v, data in skeletons.edges(data=True):
            self.assertAlmostEqual(data['length'], math.sqrt(2), places=5)

        self.assertAlmostEqual(erl, 0.5*math.sqrt(2), places=5)

    def test_merge_split_stats(self):

        skeletons = networkx.Graph()

        # one skeleton, 2*sqrt(2), no errors

        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
        }

        _, stats = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'],
            return_merge_split_stats=True)

        self.assertEqual(len(stats['merge_stats']), 0)
        self.assertEqual(len(stats['split_stats']), 0)

        # ...same, one split error

        node_segment_lut = {
            1: 10,
            2: 10,
            3: 20,
        }

        _, stats = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'],
            return_merge_split_stats=True)

        self.assertEqual(len(stats['merge_stats']), 0)
        self.assertEqual(len(stats['split_stats']), 1)
        self.assertEqual(stats['split_stats'][1], [(10, 20)])

        # one split in 1, one in 2

        skeletons = networkx.Graph()
        skeletons.add_node(1, skeleton_id=1, z=0, y=0, x=0)
        skeletons.add_node(2, skeleton_id=1, z=1, y=1, x=0)
        skeletons.add_node(3, skeleton_id=1, z=0, y=1, x=1)
        skeletons.add_edge(1, 2)
        skeletons.add_edge(2, 3)
        skeletons.add_node(4, skeleton_id=2, z=1, y=1, x=1)
        skeletons.add_node(5, skeleton_id=2, z=0, y=0, x=1)
        skeletons.add_edge(4, 5)
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 30,
            4: 20,
            5: 40
        }

        _, stats = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'],
            return_merge_split_stats=True)

        self.assertEqual(len(stats['merge_stats']), 0)
        self.assertEqual(len(stats['split_stats']), 2)
        self.assertEqual(stats['split_stats'][1], [(10, 30)])
        self.assertEqual(stats['split_stats'][2], [(20, 40)])

        # complete merge
        node_segment_lut = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 10
        }

        _, stats = evaluate.expected_run_length(
            skeletons,
            'skeleton_id',
            'length',
            node_segment_lut,
            skeleton_position_attributes=['z', 'y', 'x'],
            return_merge_split_stats=True)

        self.assertEqual(len(stats['merge_stats']), 1)
        self.assertEqual(len(stats['split_stats']), 0)
        self.assertEqual(stats['merge_stats'][10], [1, 2])
