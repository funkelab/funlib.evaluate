from funlib import evaluate
import networkx
import random
import unittest


class TestRandVoi(unittest.TestCase):

    def test_split_graph(self):

        # simple case
        #
        # 1---2---3
        #
        # split 1 / 3

        graph = networkx.Graph()
        graph.add_node(1, z=0, y=0, x=0)
        graph.add_node(2, z=1, y=1, x=0)
        graph.add_node(3, z=0, y=1, x=1)
        graph.add_edge(1, 2, weight=0)
        graph.add_edge(2, 3, weight=1)

        num_splits = evaluate.split_graph(
            graph,
            [[1], [3]],
            ['x', 'y', 'z'],
            'weight',
            'split')

        self.assertEqual(num_splits, 1)
        self.assertEqual(graph.nodes[1]['split'], 0)
        self.assertEqual(graph.nodes[2]['split'], 1)
        self.assertEqual(graph.nodes[3]['split'], 1)

        # loop
        #
        # 1---2---3
        # |       |
        # 4---5---6
        #
        # split 1 / 6

        graph = networkx.Graph()
        graph.add_node(1, z=0, y=0, x=0)
        graph.add_node(2, z=0, y=0, x=1)
        graph.add_node(3, z=0, y=0, x=2)
        graph.add_node(4, z=0, y=1, x=0)
        graph.add_node(5, z=0, y=1, x=1)
        graph.add_node(6, z=0, y=1, x=2)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=0.5)
        graph.add_edge(3, 6, weight=1)
        graph.add_edge(5, 6, weight=1)
        graph.add_edge(4, 5, weight=0.1)
        graph.add_edge(1, 4, weight=1)

        num_splits = evaluate.split_graph(
            graph,
            [[1], [6]],
            ['x', 'y', 'z'],
            'weight',
            'split')

        self.assertEqual(num_splits, 1)
        self.assertEqual(graph.nodes[1]['split'], 0)
        self.assertEqual(graph.nodes[2]['split'], 0)
        self.assertEqual(graph.nodes[4]['split'], 0)
        self.assertEqual(graph.nodes[3]['split'], 1)
        self.assertEqual(graph.nodes[5]['split'], 1)
        self.assertEqual(graph.nodes[6]['split'], 1)

        # loop, three components
        #
        # 1---2-*-3
        # |       |
        # 4-*-5---6
        #
        # split 1 / 4 / 6

        graph = networkx.Graph()
        graph.add_node(1, z=0, y=0, x=0)
        graph.add_node(2, z=0, y=0, x=1)
        graph.add_node(3, z=0, y=0, x=2)
        graph.add_node(4, z=0, y=1, x=0)
        graph.add_node(5, z=0, y=1, x=1)
        graph.add_node(6, z=0, y=1, x=2)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=0.5)
        graph.add_edge(3, 6, weight=1)
        graph.add_edge(5, 6, weight=1)
        graph.add_edge(4, 5, weight=0.1)
        graph.add_edge(1, 4, weight=1)

        num_splits = evaluate.split_graph(
            graph,
            [[1], [4], [6]],
            ['x', 'y', 'z'],
            'weight',
            'split')

        self.assertEqual(num_splits, 2)
        self.assertEqual(graph.nodes[1]['split'], 0)
        self.assertEqual(graph.nodes[2]['split'], 0)
        self.assertEqual(graph.nodes[4]['split'], 1)
        self.assertEqual(graph.nodes[3]['split'], 2)
        self.assertEqual(graph.nodes[5]['split'], 2)
        self.assertEqual(graph.nodes[6]['split'], 2)

        # loop, three components, several nodes per component
        #
        # 1---2-*-3
        # |       |
        # 4-*-5---6
        #
        # split 1 / 4 / 5,6

        graph = networkx.Graph()
        graph.add_node(1, z=0, y=0, x=0)
        graph.add_node(2, z=0, y=0, x=1)
        graph.add_node(3, z=0, y=0, x=2)
        graph.add_node(4, z=0, y=1, x=0)
        graph.add_node(5, z=0, y=1, x=1)
        graph.add_node(6, z=0, y=1, x=2)
        graph.add_edge(1, 2, weight=1)
        graph.add_edge(2, 3, weight=0.5)
        graph.add_edge(3, 6, weight=1)
        graph.add_edge(5, 6, weight=1)
        graph.add_edge(4, 5, weight=0.1)
        graph.add_edge(1, 4, weight=1)

        num_splits = evaluate.split_graph(
            graph,
            [[1], [4], [5, 6]],
            ['x', 'y', 'z'],
            'weight',
            'split')

        self.assertEqual(num_splits, 2)
        self.assertEqual(graph.nodes[1]['split'], 0)
        self.assertEqual(graph.nodes[2]['split'], 0)
        self.assertEqual(graph.nodes[4]['split'], 1)
        self.assertEqual(graph.nodes[3]['split'], 2)
        self.assertEqual(graph.nodes[5]['split'], 2)
        self.assertEqual(graph.nodes[6]['split'], 2)

        # loop, three components, several nodes per component, non-consecutive
        # IDs
        #
        # 10---20-*-30
        # |         |
        # 40-*-50---60
        #
        # split 1 / 4 / 5,6

        graph = networkx.Graph()
        graph.add_node(10, z=0, y=0, x=0)
        graph.add_node(20, z=0, y=0, x=1)
        graph.add_node(30, z=0, y=0, x=2)
        graph.add_node(40, z=0, y=1, x=0)
        graph.add_node(50, z=0, y=1, x=1)
        graph.add_node(60, z=0, y=1, x=2)
        graph.add_edge(10, 20, weight=1)
        graph.add_edge(20, 30, weight=0.5)
        graph.add_edge(30, 60, weight=1)
        graph.add_edge(50, 60, weight=1)
        graph.add_edge(40, 50, weight=0.1)
        graph.add_edge(10, 40, weight=1)

        num_splits = evaluate.split_graph(
            graph,
            [[10], [40], [50, 60]],
            ['x', 'y', 'z'],
            'weight',
            'split')

        self.assertEqual(num_splits, 2)
        self.assertEqual(graph.nodes[10]['split'], 0)
        self.assertEqual(graph.nodes[20]['split'], 0)
        self.assertEqual(graph.nodes[40]['split'], 1)
        self.assertEqual(graph.nodes[30]['split'], 2)
        self.assertEqual(graph.nodes[50]['split'], 2)
        self.assertEqual(graph.nodes[60]['split'], 2)

        # chain with several components, increasing edge weights
        #
        # 1--2--3--4--5--6--66--7--8--9 nodes
        # 1  2  1  1  2  3  3   4  4  4 components

        graph = networkx.Graph()
        graph.add_node(1, x=1)
        graph.add_node(2, x=2.1)  # move closer to 3
        graph.add_node(3, x=3)
        graph.add_node(4, x=4)
        graph.add_node(5, x=5)
        graph.add_node(6, x=6)
        graph.add_node(66, x=6.5)
        graph.add_node(7, x=7)
        graph.add_node(8, x=8)
        graph.add_node(9, x=9)
        graph.add_edge(1, 2, weight=0.1)
        graph.add_edge(2, 3, weight=0.2)
        graph.add_edge(3, 4, weight=0.3)
        graph.add_edge(4, 5, weight=0.4)
        graph.add_edge(5, 6, weight=0.5)
        graph.add_edge(6, 66, weight=0.55)
        graph.add_edge(66, 7, weight=0.6)
        graph.add_edge(7, 8, weight=0.7)
        graph.add_edge(8, 9, weight=0.8)

        num_splits = evaluate.split_graph(
            graph,
            [[1, 3, 4], [2, 5], [6, 66], [7, 8, 9]],
            ['x'],
            'weight',
            'split')

        # should first split [1, 3, 4] and [7, 8, 9] at 4/7
        #
        # 1--2--3--4  5--6--66--7--8--9 nodes
        # 1  2  1  1  2  3  3   4  4  4 components
        #                               split label
        #
        # then split [1, 3, 4] and [2] at 1/2
        #
        # 1  2--3--4  5--6--66--7--8--9 nodes
        # 1  2  1  1  2  3  3   4  4  4 components
        # 0                             split label
        #
        # then split [3, 4] and [2] at 2/3
        #
        # 1  2  3--4  5--6--66--7--8--9 nodes
        # 1  2  1  1  2  3  3   4  4  4 components
        # 0  1  2  2                    split label
        #
        # then split [6, 66] and [7, 8, 9] at 66/7
        #
        # 1  2  3--4  5--6--66  7--8--9 nodes
        # 1  2  1  1  2  3  3   4  4  4 components
        # 0  1  2  2                    split label
        #
        # then split [5] and [6, 66] at 5/6
        #
        # 1  2  3--4  5  6--66  7--8--9 nodes
        # 1  2  1  1  2  3  3   4  4  4 components
        # 0  1  2  2  3  4  4   5  5  5 split label
        #
        # and done
        self.assertEqual(num_splits, 5)
        self.assertEqual(graph.nodes[1]['split'], 0)
        self.assertEqual(graph.nodes[2]['split'], 1)
        self.assertEqual(graph.nodes[3]['split'], 2)
        self.assertEqual(graph.nodes[4]['split'], 2)
        self.assertEqual(graph.nodes[5]['split'], 3)
        self.assertEqual(graph.nodes[6]['split'], 4)
        self.assertEqual(graph.nodes[66]['split'], 4)
        self.assertEqual(graph.nodes[7]['split'], 5)
        self.assertEqual(graph.nodes[8]['split'], 5)
        self.assertEqual(graph.nodes[9]['split'], 5)

        # random graph

        graph = networkx.Graph()
        random.seed(2)
        num_nodes = 1000
        num_edges = 10000
        for i in range(num_nodes):
            graph.add_node(
                i,
                z=random.random(),
                y=random.random(),
                x=random.random())
        for i in range(num_edges):
            graph.add_edge(
                random.randint(0, num_nodes - 1),
                random.randint(0, num_nodes - 1),
                weight=random.random())

        component_a = list(range(0, num_nodes//100))
        component_b = list(range(num_nodes//5, num_nodes//5 + num_nodes//100))
        num_splits = evaluate.split_graph(
            graph,
            [component_a, component_b],
            ['x', 'y', 'z'],
            'weight',
            'split')

        self.assertEqual(num_splits, 18)
