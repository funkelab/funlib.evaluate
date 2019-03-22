from scipy.spatial import KDTree
import graph_tool
import graph_tool.flow
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

    # number nodes consecutively in graph
    for i, (_, data) in enumerate(graph.nodes(data=True)):
        data['_gt_id'] = i

    # map components to positions
    component_node_positions = {
        graph.nodes[n]['_gt_id']: np.array([
            graph.nodes[n][p] for p in position_attributes
        ])
        for component in components
        for n in component
    }

    # replace component node IDs with gt graph IDs
    components = [
        [graph.nodes[n]['_gt_id'] for n in component]
        for component in components
    ]

    # create edge list
    edges = np.array([
        [graph.nodes[u]['_gt_id'], graph.nodes[v]['_gt_id']]
        for u, v in graph.edges()])

    # create weights list
    weights = np.array(
        [d[weight_attribute] for _, _, d in graph.edges(data=True)],
        dtype=np.float32)

    # create graph_tool graph
    gt_graph = graph_tool.Graph()
    gt_graph.add_edge_list(edges)
    gt_graph.add_edge_list(edges[:, [1, 0]])  # edges are directed
    weights = gt_graph.new_edge_property(
        'double',
        np.concatenate([weights, weights]))
    split_labels = gt_graph.new_vertex_property('int64_t', val=-1)

    # recursively split the graph
    num_splits = rec_split_graph(
        gt_graph,
        weights,
        components,
        component_node_positions,
        split_labels,
        0)[0]

    for i, (_, data) in enumerate(graph.nodes(data=True)):
        data[split_attribute] = split_labels[i]

    return num_splits


def rec_split_graph(
        graph,
        weights,
        components,
        component_node_positions,
        split_labels,
        next_split_id):

    # nothing to split?
    if len(components) <= 1:
        for node in graph.vertices():
            split_labels[node] = next_split_id
        return 0, next_split_id + 1

    # find split nodes
    component_u, component_v = select_split_component(components)
    u, v = select_split_nodes(
        component_u,
        component_v,
        component_node_positions)

    # split graph into split_u and split_v
    split_u, split_v = min_cut(graph, u, v, weights)

    component_nodes_u = filter_component_nodes(split_u, components)
    component_nodes_v = filter_component_nodes(split_v, components)

    # split recursively
    num_splits_needed_u, next_split_id = rec_split_graph(
        split_u,
        weights,
        component_nodes_u,
        component_node_positions,
        split_labels,
        next_split_id)
    num_splits_needed_v, next_split_id = rec_split_graph(
        split_v,
        weights,
        component_nodes_v,
        component_node_positions,
        split_labels,
        next_split_id)

    return 1 + num_splits_needed_u + num_splits_needed_v, next_split_id


def select_split_component(components):

    # return largest two components
    components = sorted(components, key=lambda l: len(l))
    return components[-2:]


def select_split_nodes(
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


def min_cut(graph, u, v, weights):

    res = graph_tool.flow.boykov_kolmogorov_max_flow(
        graph,
        graph.vertex(u), graph.vertex(v),
        weights)

    partition = graph_tool.flow.min_st_cut(
        graph,
        u,
        weights,
        res)

    split_u = graph_tool.GraphView(graph, vfilt=partition)
    split_v = graph_tool.GraphView(graph, vfilt=np.logical_not(partition.a))

    return split_u, split_v


def filter_component_nodes(graph, components):
    '''Return a list of component nodes limited to nodes in graph.'''

    # filter nodes
    components = [
        [
            n
            for n in comp
            if n in graph.vertices()
        ]
        for comp in components
    ]

    # remove empty lists
    components = [c for c in components if len(c) > 0]

    return components
