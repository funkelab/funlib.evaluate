from scipy.spatial import KDTree
import networkx
import numpy as np


def split_graph(
        graph,
        components,
        position_attributes,
        weight_attribute,
        split_attribute):
    '''Recursively split a graph via min-cuts such that the given component
    nodes are separated.

    In every iteration, the two largest components that are not split are
    selected, and a min-cut is sought between the two closest nodes of these
    two components.

    Args:

        graph (networkx-like graph):

            The graph to split.

        components (``list`` of ``list`` of nodes):

            The nodes to separate from each other.

        position_attributes (``list`` of ``string``):

            The names of the spatial position attributes of the nodes. Needed
            to find the closest split nodes.

        weight_attribute (``string``):

            The name of the edge attribute that holds the weight for the
            min-cut.

        split_attribute (``string``):

            The name of the node attribute in which to store the ID of the
            split each node ended up in.

    Returns:

        The number of splits performed.
    '''

    component_node_positions = {
        n: np.array([
            graph.nodes[n][p] for p in position_attributes
        ])
        for component in components
        for n in component
    }

    return rec_split_graph(
        graph,
        components,
        component_node_positions,
        weight_attribute,
        split_attribute,
        0)[0]


def rec_split_graph(
        graph,
        components,
        component_node_positions,
        weight_attribute,
        split_attribute,
        next_split_id):

    # nothing to split?
    if len(components) <= 1:
        for node, data in graph.nodes(data=True):
            data[split_attribute] = next_split_id
        return 0, next_split_id + 1

    # find split nodes
    component_u, component_v = select_split_component(components)
    u, v = select_split_nodes(
        graph,
        component_u,
        component_v,
        component_node_positions)

    # split graph into split_u and split_v
    split_u, split_v = min_cut(graph, u, v, weight_attribute)

    component_nodes_u = filter_component_nodes(split_u, components)
    component_nodes_v = filter_component_nodes(split_v, components)

    # split recursively
    num_splits_needed_u, next_split_id = rec_split_graph(
        split_u,
        component_nodes_u,
        component_node_positions,
        weight_attribute,
        split_attribute,
        next_split_id)
    num_splits_needed_v, next_split_id = rec_split_graph(
        split_v,
        component_nodes_v,
        component_node_positions,
        weight_attribute,
        split_attribute,
        next_split_id)

    # copy split labels
    for node, data in split_u.nodes(data=True):
        graph.nodes[node][split_attribute] = data[split_attribute]
    for node, data in split_v.nodes(data=True):
        graph.nodes[node][split_attribute] = data[split_attribute]

    return 1 + num_splits_needed_u + num_splits_needed_v, next_split_id


def select_split_component(components):

    # return largest two components
    components = sorted(components, key=lambda l: len(l))
    return components[-2:]


def select_split_nodes(
        graph,
        component_u,
        component_v,
        component_node_positions):
    '''Find the two spatially closest component nodes.'''

    kd_tree_u = KDTree(
        [component_node_positions[n] for n in component_u]
    )
    distances, indices = kd_tree_u.query(
        [component_node_positions[n] for n in component_v]
    )

    v_index = np.argmin(distances)
    u_index = indices[v_index]

    return component_u[u_index], component_v[v_index]


def min_cut(graph, u, v, weight_attribute):

    value, partition = networkx.minimum_cut(
        graph,
        u,
        v,
        capacity=weight_attribute)

    split_u = graph.copy()
    split_v = graph.copy()

    split_u.remove_nodes_from(partition[1])
    split_v.remove_nodes_from(partition[0])

    return split_u, split_v


def filter_component_nodes(graph, components):
    '''Return a list of component nodes limited to nodes in graph.'''

    # filter nodes
    components = [
        [
            n
            for n in comp
            if n in graph
        ]
        for comp in components
    ]

    # remove empty lists
    components = [c for c in components if len(c) > 0]

    return components
