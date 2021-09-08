import numpy as np
import networkx as nx
from networkx import Graph, adjacency_matrix
import tensorflow as tf
from typing import Union

def a2g(A):
    """
    Adjacency matrix to graph object.
    Args:
        A: A ndarray, binary adjacency matrix.

    Returns:
        g: A nxgraph, networkx graph object.
    """
    g = nx.Graph()
    a = np.where(A == 1)
    a = [[x, y] for x, y in zip(a[0], a[1])]
    g.add_edges_from(a)
    return g

def renormalization(a):
    """
    Give an adjacency matrix and returns the renormalized.
    Args:
        a: A ndarray, adjacency matrix.

    Returns:
        atld: A ndarray, renormalized adjacency matrix.

    Examples:
        >>> a = np.array([[[0,1,1], [1,0,0], [1,0,0]]])
        >>> atld = renormalization(a)
        >>> print(atld)
        [[[0.33333333 0.         0.        ]
          [0.         0.5        0.        ]
          [0.         0.         0.5       ]]]

    References:
        Thomas N. Kipf, Max Welling. Semi-supervised classification with graph convolutional networks,
        https://arxiv.org/pdf/1609.02907.pdf
    """

    ai = a + np.eye(a.shape[-1])
    degree = np.sum(ai, axis=-1)
    degree = np.eye(a.shape[-1]) * degree
    degree_inv = np.linalg.inv(degree)
    degree_inv = np.power(degree_inv, 0.5)

    atld = np.matmul(degree_inv, ai)
    atld = np.matmul(atld, degree_inv)
    return atld

def random_uncycle_graph(num_v:int):
    """
    Given a number of vertices returns a random uncycle graph with range of vertices: [number_of_vertices, 2 * number_of_vertices].
    Args:
        num_v: A int, number of vertices

    Returns:
        A: A ndarray, adjacency matrix of random uncycle graph,
        Atld: A ndarray, normilized (Laplacian normalization) adjacency matrix of $A$,
        X: A ndarray, feature matrix, same size as $A$.
    """
    g = nx.generators.random_graphs.dense_gnm_random_graph(num_v, num_v * 2)
    while True:
        try:
            edges = nx.cycles.find_cycle(g)
        except:
            break
        g.remove_edge(*edges[0])
    A = nx.adjacency_matrix(g).toarray()
    Atld = renormalization(A)
    X = np.eye(A.shape[1])
    X = X.reshape(X.shape[0], 1)
    return A, Atld, X


def gen_batch_graphs(n_graphs=500, min_nv=5, max_nv=8):
    """
    Get disjoint matrix of a given number of graphs with random range [$min_nv$, $max_nv$]. The outcome is disjoint matrix
    created by random uncycle graph.

    Args:
        n_graphs: A int (optional), number of generated graphs. Default is 500.
        min_nv: A int (optional), the minimum number of vertices. Default is 5.
        max_nv: A int (optional), the maximum number of vertices. Default is 8.

    Returns:
        A: A ndarray, disjoint adjacency matrix
        Atld: A ndarray, renormalized disjoint adjacency matrix of $A$.
        X: A ndarray, identity matrix. Same size as $A$.
    """
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(min_nv, max_nv, n_graphs)
        for num_v in gen_v:
            A, Atld, X = random_uncycle_graph(num_v)
            A_list.append(A)
            Atld_list.append(Atld)
            X_list.append(X)
        X, A, I = numpy_to_disjoint(X_list, A_list)
        _, Atld, _ = numpy_to_disjoint(X_list, Atld_list)
        A = A.toarray()
        Atld = Atld.toarray()
        yield A, Atld, X


def nx_graph(num_v, want):
    """Random Graph.

    Args:
        num_v: A int, Number of vertexes
        want: A list, Type graphs which want to generate

    Returns:
        A: A ndarray, Adjacency matrix
        Atld: A ndarray, Normalized adjacency matrix
        X: A ndarray, Nodes Features (identity matrix)
        rgi: A int, Random graph index
    """
    w_sum = sum(want)
    want = [sum(want[:i]) if want[i] == 1 else -1 for i in range(len(want))]
    rgi = np.random.randint(0, w_sum)
    if rgi == want[0]:
        g = nx.cycle_graph(num_v)
    elif rgi == want[1]:
        g = nx.star_graph(num_v - 1)
    elif rgi == want[2]:
        g = nx.wheel_graph(num_v)
    elif rgi == want[3]:
        g = nx.complete_graph(num_v)
    elif rgi == want[4]:
        path_len = np.random.randint(2, num_v // 2)
        g = nx.lollipop_graph(m=num_v - path_len, n=path_len)
    elif rgi == want[5]:
        g = nx.hypercube_graph(np.log2(num_v).astype('int'))
        g = nx.convert_node_labels_to_integers(g)
    elif rgi == want[6]:
        g = nx.circular_ladder_graph(num_v // 2)
    elif rgi == want[7]:
        n_rows = np.random.randint(2, num_v // 2)
        n_cols = num_v // n_rows
        g = nx.grid_graph([n_rows, n_cols])
        g = nx.convert_node_labels_to_integers(g)

    A = nx.adjacency_matrix(g)
    Atld = renormalization(A)
    X = np.eye(A.shape[0])
    return A, Atld, X, rgi


def numpy_to_mega_batch(X:list, A:list):
    """
    Args:
        X: A list, list of feature matrixes,
        A: A list, list of adjency matrixes.
    Examples:
        >>> X = [np.random.rand(6,4) for _ in range(12)]
        >>> A = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
        >>> X, A = numpy_to_mega_batch(X,A)
        >>> print(A.shape)
        (12, 6, 6)
        >>> print(X.shape)
        (12, 6, 4)
    """
    def post_concat(x):
        """
        Post concatenation at 1st and 2nd dimension of array.
        Args:
            x: A ndarray, minimum 3D array.

        Returns:
            x_con_post: A ndarray, post concatenation array. Minimum 3D array.

        Examples:
            >>> x = np.array([[[1,1], [2,2]]])
            >>> x_con_post = post_concat(x)
            >>> print(x_con_post)

        """
        x_con_post = np.concatenate([x, np.zeros((x.shape[0], max_d-x.shape[1]))], axis=1)
        x_con_post = np.concatenate([x_con_post, np.zeros((max_d - x_con_post.shape[0], x_con_post.shape[1]))], axis=0)
        return x_con_post

    max_d = max([a.shape[0] for a in A])
    mega_batch_A = []
    mega_batch_X = []
    for (x, a) in zip(X, A):
        if a.shape[0] < max_d:
            a = post_concat(a)
            x = post_concat(x)
        mega_batch_A.append(a)
        mega_batch_X.append(x)
    mega_batch_A = np.array(mega_batch_A)
    mega_batch_X = np.stack(mega_batch_X, axis=0)
    return  mega_batch_X, mega_batch_A


def numpy_to_disjoint(X:list, A:list):
    """
    Args:
        X: a list, list of np feature matrixes,
        A: a list, list of np adajence matrixes.

    Returns:
        disjoint_X: A ndarray, np disjont matrix of list of X argument,
        disjoint_A: A ndarray, np disjont matrix of list of A argument.

    Examples:
        >>> X = [np.random.rand(6,4) for _ in range(12)]
        >>> A = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
        >>> X, A = numpy_to_disjoint(X,A)
        >>> print(A.shape)
        (54, 54)
        >>> print(X.shape)
        (12, 6, 4)
    """
    I = []
    disjoint_A = A[0]
    for a in A[1:]:
        na = a.shape[1]
        ndA = disjoint_A.shape[1]
        disjoint_A = np.concatenate([disjoint_A, np.zeros((disjoint_A.shape[0], na))], axis=1)
        a = np.concatenate([np.zeros(( a.shape[0], ndA)), a], axis=1)
        disjoint_A = np.concatenate([disjoint_A, a], axis=0)
    disjoint_X = np.stack(X,axis=0)
    return  disjoint_X, disjoint_A

def gen_nx_graphs(batch=100, want=[1, 1, 1, 1, 1, 1, 1, 1], minN=6, maxN=10):
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        C_list = []
        gen_v = np.random.randint(minN, maxN, batch)
        maxN = 0
        for num_v in gen_v:
            A, Atld, X, C = nx_graph(num_v, want)
            if maxN < A.shape[0]:
                maxN = A.shape[0]
            A_list.append(A)

            Atld_list.append(Atld)
            X_list.append(X)

            C_list.append(C)

        for i in range(len(X_list)):
            x = X_list[i]
            left_pad = np.zeros((x.shape[0], maxN - x.shape[1]))
            bottom_pad = np.zeros((maxN - x.shape[0], maxN))

            x = np.concatenate([x, left_pad], axis=1)
            x = np.concatenate([x, bottom_pad], axis=0)
            X_list[i] = x

        X, A = numpy_to_mega_batch(X_list, A_list)
        _, Atld = numpy_to_mega_batch(X_list, Atld_list)
        A = np.array(A)
        Atld = np.array(Atld)
        yield X, A, Atld, C_list


def gen_nx_random_uncycle_graphs():
    batch = 1000
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(5, 10, batch)
        for num_v in gen_v:
            r_choice = np.random.randint(2)
            if r_choice == 0:
                A, Atld, X = nx_graph(num_v)
            else:
                A, Atld, X = random_uncycle_graph(num_v)
            A_list.append(A)
            Atld_list.append(Atld)
            X_list.append(X)

        X, A = numpy_to_mega_batch(X_list, A_list)
        _, Atld = numpy_to_mega_batch(X_list, Atld_list)
        A = A.toarray()
        Atld = Atld.toarray()
        yield A, Atld, X


def nodelist2graph(nodelist:list, maxN=None):
    """
    A list of sublist with numbering nodes of graph. Output is a tf_tensor of batch Adjacency matrix per graph.
    Args:
        nodelist: A list, nested list with nodes of the graphs,
        maxN: An int (optional), max number of nodes or max size of Adjacency matrix. Default is None.

    Returns:
        X: A tf_tensor, feature matrix per graph,
        A: A tf_tensor, Adjacency matrix per graph,
        Atld: A tf_tensor, normalized Adjacency matrix.

    Examples:
        >>> node_list = [[[0,1], [1,2], [1,3], [1,4], [1,5]], [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]]
        >>> X, A, Atld = nodelist2graph(node_list)
        >>> print(X)
        >>> print(A)
        >>> print(Atld)
    """
    A_list = []
    Atld_list = []
    X_list = []

    for elements in nodelist:
        g = Graph(elements)
        a = adjacency_matrix(g, sorted(g.nodes())).toarray()
        a_tld = renormalization(a)
        x = np.eye(a.shape[0])
        A_list.append(a)
        Atld_list.append(a_tld)
        X_list.append(x)

    X, A = numpy_to_mega_batch(X_list, A_list)
    _, Atld = numpy_to_mega_batch(X_list, Atld_list)

    X = tf.cast(X, tf.float32)
    A = tf.cast(A, tf.float32)
    Atld = tf.cast(Atld, tf.float32)

    if maxN is None:
        return X, A, Atld
    elif maxN <= X.shape[1]:
        X = X[:, :maxN, :maxN]
        A = A[:, :maxN, :maxN]
        Atld = Atld[:, :maxN, :maxN]
        return X, A, Atld
    else:
        left_pad = tf.zeros((X.shape[0], X.shape[1], maxN - X.shape[2]))
        bottom_pad = tf.zeros((X.shape[0], maxN - X.shape[1], maxN))

        X = tf.concat([X, left_pad], axis=2)
        X = tf.concat([X, bottom_pad], axis=1)

        A = tf.concat([A, left_pad], axis=2)
        A = tf.concat([A, bottom_pad], axis=1)

        Atld = tf.concat([Atld, left_pad], axis=2)
        Atld = tf.concat([Atld, bottom_pad], axis=1)
        return X, A, Atld


def graphs_stats(A: np.ndarray) -> Union[int, dict, dict, dict]:
    """
    Statistics about graph examples. Run bfs tree on every node on every graph getting the sortest path lengest per node
    (depth). After that mesure the max depth, depth distribution per depth observation, max depth distribution (geting
    the maximum depth per graph and update the distribution) and the number of edges distribution for all nodes for all graphs.

    Args:
        A: A ndarray, a list of adjacency matrix.

    Returns:
        max_depth: A int, max depth of all graphs for all nodes of bfs tree,
        depth_dist: A dict, keys are the observable depth of bfs tree (such as 1, 2, 4, 5, ...),
        maxdepth_dist: A dict, keys are the observable max-depth per node of bfs tree (such as 1, 2, 4, 5, ...),
        edge_n: A dict, keys are the observable number of edges per graph.
    """
    max_depth = 0
    depth_dist = {}
    maxdepth_dist = {}
    edge_n = {}
    for a in A:
        g = a2g(a)
        en = len(g.edges())
        if en not in edge_n.keys():
            edge_n[en] = 1
        else:
            edge_n[en] += 1
        for n in g.nodes():
            tree = nx.algorithms.traversal.breadth_first_search.bfs_tree(g, n)
            depths = nx.shortest_path_length(tree, n).values()
            val = 1
            for d in depths:
                if d > -1:
                    if d not in depth_dist.keys():
                        depth_dist[d] = val
                    else:
                        depth_dist[d] += val

            maxd = max(depths)
            if maxd not in maxdepth_dist.keys():
                maxdepth_dist[maxd] = val
            else:
                maxdepth_dist[maxd] += val
            if maxd > max_depth:
                max_depth = maxd
    return max_depth, depth_dist, maxdepth_dist, edge_n