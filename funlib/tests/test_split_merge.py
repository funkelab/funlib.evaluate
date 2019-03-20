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
