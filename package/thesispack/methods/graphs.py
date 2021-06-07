import numpy as np
import networkx as nx
from networkx import Graph, adjacency_matrix
import tensorflow as tf
from spektral.utils import gcn_filter


def A2G(A):
    g = nx.Graph()
    a = np.where(A == 1)
    a = [[x, y] for x, y in zip(a[0], a[1])]
    g.add_edges_from(a)
    return g


def random_uncycle_graph(num_v):
    g = nx.generators.random_graphs.dense_gnm_random_graph(num_v, num_v * 2)
    while True:
        try:
            edges = nx.cycles.find_cycle(g)
        except:
            break
        g.remove_edge(*edges[0])
    A = nx.adjacency_matrix(g).toarray()
    A = A + np.eye(A.shape[1])
    Atld = gcn_filter(A)
    X = np.eye(A.shape[1])
    X = X.reshape(X.shape[0], 1)
    return A, Atld, X


def gen_batch_graphs():
    batch = 500
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(5, 8, batch)
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
    A = A + np.eye(A.shape[1])
    Atld = gcn_filter(A)
    X = np.eye(A.shape[0])
    return A, Atld, X, rgi


def numpy_to_mega_batch(X:list, A:list):
    """
    >>> X = [np.random.rand(6,4) for _ in range(12)]
    >>> A = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
    >>> X, A = numpy_to_mega_batch(X,A)
    >>> assert A.shape == (12, 6, 6)
    >>> assert X.shape == (12, 6, 4)
    """
    max_d = max([a.shape[0] for a in A])
    mega_batch_A = []
    mega_batch_X = []
    for (x, a) in zip(X, A):
        if a.shape[0] < max_d:
            a = np.concatenate([a, np.zeros((a.shape[0], max_d-a.shape[1]))], axis=1)
            a = np.concatenate([a, np.zeros((max_d-a.shape[0], a.shape[1]))], axis=0)
        mega_batch_A.append(a)
        mega_batch_X.append(x)
    mega_batch_A = np.array(mega_batch_A)
    mega_batch_X = np.stack(X,axis=0)
    return  mega_batch_X, mega_batch_A


def numpy_to_disjoint(X:list, A:list):
    """
    >>> X = [np.random.rand(6,4) for _ in range(12)]
    >>> A = [np.random.rand(6,6) for _ in range(6)]+[np.random.rand(3,3) for _ in range(6)]
    >>> X, A = numpy_to_disjoint(X,A)
    >>> assert A.shape == (54, 54)
    >>> assert X.shape == (12, 6, 4)
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


def list2graph(nodelist, maxN=None):
    A_list = []
    Atld_list = []
    X_list = []

    for elements in nodelist:
        g = Graph(elements)
        a = adjacency_matrix(g, sorted(g.nodes())).toarray()
        a = a + np.eye(a.shape[1])
        a_tld = gcn_filter(a)
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


def graphs_stats(Adjs):
    maxD = 0
    depth_dist = {}
    maxDs = {}
    edgesN = {}
    for a in Adjs:
        g = A2G(a)
        en = len(g.edges())
        if en not in edgesN.keys():
            edgesN[en] = 1
        else:
            edgesN[en] += 1
        for n in g.nodes():
            tree = nx.algorithms.traversal.breadth_first_search.bfs_tree(g, n)
            depths = nx.shortest_path_length(tree, n).values()
            # val = 1/len(depths)
            val = 1
            for d in depths:
                if d > -1:
                    if d not in depth_dist.keys():
                        depth_dist[d] = val
                    else:
                        depth_dist[d] += val

            maxd = max(depths)
            if maxd not in maxDs.keys():
                maxDs[maxd] = val
            else:
                maxDs[maxd] += val
            if maxd > maxD:
                maxD = maxd
    return maxD, depth_dist, maxDs, edgesN


if __name__ == "__main__":
    import doctest
    doctest.testmod()