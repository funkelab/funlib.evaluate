from scipy.spatial import cKDTree as KDTree
import graph_tool
import graph_tool.flow
import numpy as np
import time


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

    # split graph
    partition = min_cut(graph, u, v, weights)

    # split recursively

    prev_partition = graph.get_vertex_filter()[0]

    graph.set_vertex_filter(partition)
    component_nodes_u = filter_component_nodes(graph, components)
    num_splits_needed_u, next_split_id = rec_split_graph(
        graph,
        weights,
        component_nodes_u,
        component_node_positions,
        split_labels,
        next_split_id)

    if prev_partition:
        partition.a = np.logical_and(
            np.logical_not(partition.a),
            prev_partition.a)
    else:
        partition.a = np.logical_not(partition.a)

    graph.set_vertex_filter(partition)
    component_nodes_v = filter_component_nodes(graph, components)
    num_splits_needed_v, next_split_id = rec_split_graph(
        graph,
        weights,
        component_nodes_v,
        component_node_positions,
        split_labels,
        next_split_id)

    graph.clear_filters()

    return 1 + num_splits_needed_u + num_splits_needed_v, next_split_id


def select_split_component(components):

    start = time.time()

    # return largest two components
    components = sorted(components, key=lambda l: len(l))

    print("Found split components in %.3fs" % (time.time() - start))
    return components[-2:]


def select_split_nodes(
        component_u,
        component_v,
        component_node_positions):
    '''Find the two spatially closest component nodes.'''

    start = time.time()

    kd_tree_u = KDTree(
        [component_node_positions[n] for n in component_u]
    )
    distances, indices = kd_tree_u.query(
        [component_node_positions[n] for n in component_v]
    )

    v_index = np.argmin(distances)
    u_index = indices[v_index]

    print("Found split nodes in %.3fs" % (time.time() - start))
    return component_u[u_index], component_v[v_index]


def min_cut(graph, u, v, weights):

    start = time.time()

    res = graph_tool.flow.boykov_kolmogorov_max_flow(
        graph,
        graph.vertex(u), graph.vertex(v),
        weights)

    partition = graph_tool.flow.min_st_cut(
        graph,
        u,
        weights,
        res)

    print("Split in %.3fs" % (time.time() - start))
    return partition


def filter_component_nodes(graph, components):
    '''Return a list of component nodes limited to nodes in graph.'''

    start = time.time()

    vertex_filter, inverted = graph.get_vertex_filter()
    vertex_filter_array = vertex_filter.a
    contained = not inverted

    # filter nodes
    components = [
        [
            n
            for n in comp
            if vertex_filter_array[n] == contained
        ]
        for comp in components
    ]

    # remove empty lists
    components = [c for c in components if len(c) > 0]

    print("Filtered components in %.3fs" % (time.time() - start))
    return components
