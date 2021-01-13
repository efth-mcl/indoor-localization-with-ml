import os
import subprocess
import json
import numpy as np
import tensorflow as tf
import time
from spektral.utils import localpooling_filter
from spektral.utils.data import numpy_to_batch
from networkx import Graph, adjacency_matrix
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product


class Iperf3Out:
    
    def __init__(self, **entries):
        self.__dict__.update(entries)


def iperf3_udp_throughput(c, B, p, t=30, b_Mbps=32, R=False, zero_copy=False):
    iperf3cmd = 'iperf3 -c {} -B {} -p {} -u -t {} -b {}Mbps'.format(c, B, p, t, b_Mbps)
    if R:
        iperf3cmd+= ' -R'
    if zero_copy:
        iperf3cmd+= ' -Z'
    iperf3cmd+= ' -J'

    iperf3cmd = iperf3cmd.split()
    iperf3cmd_out = subprocess.run(iperf3cmd,stdout=subprocess.PIPE)
    iperf3cmd_out = iperf3cmd_out.stdout.decode('utf-8')

    iperf3out2json = json.loads(iperf3cmd_out)
    return iperf3out2json


def iperf3_udp_obj_custom_avg(udp_obj):
    Mbps = []
    for inter in udp_obj.intervals:   
        Mbps.append(   
            inter['sum']['bits_per_second']/(10**6)
        )
    return np.mean(Mbps)

def udp_figure(udpresults_dict,figsize=(16,8),ylim_max=23,legend_fontsize=20,legend_loc='upper left', 
    axes_label_fontsize=20,ticks_fontsize=16,plot_icons=['*-','o-','^-'],markersize=10,save_obj=None):
    plt.figure(figsize=figsize)
    for (key, value), plot_icon  in zip(udpresults_dict.items(), plot_icons):
        udptime = [t for t in range(1, len(value)+1)]
        plt.plot(udptime, value ,plot_icon,markersize=markersize,label=key)

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.xlabel(r'$time \, (s)$',fontsize=axes_label_fontsize)
    plt.ylabel(r'$UDP \, Throughput \, (Mbps)$', fontsize=axes_label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    
    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0,ylim_max)
    
    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')
    

def mean_udp_per_dBm_bar_figure(avgsdBm_df, figsize=(16,8),ylim_max=23, 
    legend_fontsize=20,legend_loc='upper left', axes_label_fontsize=20, 
    ticks_fontsize=20,save_obj=None):

    plt.figure(figsize=figsize)
    index = np.arange(3)
    bar_width = 0.15
    for i, key in enumerate(avgsdBm_df.keys()):
        plt.bar(index+bar_width*i, avgsdBm_df[key], bar_width,label=key)
    
    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.ylabel(r'$Mean \, UDP \, Throughput \, (Mbps)$',
        fontsize=axes_label_fontsize)

    plt.xticks(index + bar_width, [r'${}$'.format(idx) for idx in avgsdBm_df.index],
        fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)

    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')


def model_feagure(means_vals, model_vals, figsize=(16,8), ylim_max=23, legend_fontsize=20,
    legend_loc='upper left', axes_label_fontsize=20, ticks_fontsize=16, plot_icons=['*-','o-'],
    markersize=10, save_obj=None):

    plt.figure(figsize=figsize)
    hopsaxes = [h for h in range(1, len(means_vals)+1)]

    plt.plot(hopsaxes, means_vals, plot_icons[0], markersize=markersize, label=r'$Mean \, UDP$')
    plt.plot(hopsaxes, model_vals, plot_icons[1], markersize=markersize, label=r'$Model \, UDP$')

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.xlabel(r'$hops \, \#$',fontsize=axes_label_fontsize)
    plt.ylabel(r'$Mean \, UDP \, Throughput \, (Mbps)$', fontsize=axes_label_fontsize)
    plt.xticks([1,2,3], fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0,ylim_max)
    
    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj), format='eps')


def mean_udp_figure(avgsdBm_df, figsize=(16,8),ylim_max=23, legend_fontsize=20,
    legend_loc='upper left', axes_label_fontsize=20, ticks_fontsize=20,save_obj=None):
    
    plt.figure(figsize=figsize)
    index = np.arange(3)
    bar_width = 0.15
    for i, key in enumerate(avgsdBm_df.keys()):
        plt.bar(index+bar_width*i, avgsdBm_df[key], bar_width,label=key)
    
    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.ylabel(r'$Mean \, UDP \, Throughput \, (Mbps)$',
        fontsize=axes_label_fontsize)

    plt.xticks(index + bar_width, [r'${}$'.format(idx) for idx in avgsdBm_df.index],
        fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)

    if save_obj is not None:
        plt.savefig('{}/eps_pics/{}/{}.eps'.format(*save_obj), format='eps')


def mean_udp_bar3d_figure(complite_means_per_txpower, line_means_per_txpower, babeld_means_per_txpower, 
    figsize=(8,8), legend_fontsize=20,legend_loc='upper right', axes_label_fontsize=12, ticks_fontsize=12,save_obj=None):

    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')

    xy = list(product([1,2,3],[1,2,3]))
    x = np.array([x[0] for x in xy])
    y = np.array([y[1] for y in xy])
    complite = complite_means_per_txpower.flatten()
    line = line_means_per_txpower.flatten()
    babeld = babeld_means_per_txpower.flatten()

    ax.bar3d(x, y, 0, [0.1]*9, [0.1]*9, complite.reshape(9), color="tab:blue", label="Complite Graph")
    cl_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:blue")
    ax.bar3d(x+0.1, y, 0, [0.1]*9, [0.1]*9, line.reshape(9), color="tab:orange", label="Line Graph")
    ln_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:orange")
    ax.bar3d(x+0.2, y, 0, [0.1]*9, [0.1]*9, babeld.reshape(9),  color="tab:green", label="Babeld")
    ba_proxy = plt.Rectangle((0, 0), 1, 1, fc="tab:green")
    ax.set_zlabel(r'$Mean \, UDP \, Throughput \, (Mbps)$')

    ax.view_init(45, 40)
    ticks = [1,2,3]
    
    plt.xticks(ticks, ['10 dBm', '20 dBm', '31 dBm'], fontsize=ticks_fontsize)
    plt.yticks(ticks, ['RPI 0 (c) -- RPI 1 (s)', 'RPI 0 (c) -- RPI 2 (s)', 'RPI 0 (c) -- RPI 3 (s)'], fontsize=ticks_fontsize)
    ax.legend([cl_proxy,ln_proxy, ba_proxy],['Complite Graph','Line Graph', 'Babeld'])
    if save_obj is not None:
        plt.savefig('DataFigures/routing/{}/{}.eps'.format(*save_obj),format='eps')


def full_mesurment_to_tf(secs_before_start_mesure=5):
    time.sleep(secs_before_start_mesure)
    # 2c -- 3s
    udp3 = iperf3_udp_throughput('10.42.0.3','10.42.0.2',6262,24,40,R=True)
    udp3 = Iperf3Out(**udp3)

    time.sleep(5)
    # 2c -- 14s
    udp14 = iperf3_udp_throughput('10.42.0.14','10.42.0.2',6262,24,40,R=True)
    udp14 = Iperf3Out(**udp14)

    time.sleep(5)
    # 2c -- 12s
    udp12 = iperf3_udp_throughput('10.42.0.12','10.42.0.2',6262,24,40,R=True)
    udp12 = Iperf3Out(**udp12)

    time.sleep(5)
    # 2c -- 7s
    udp7 = iperf3_udp_throughput('10.42.0.7','10.42.0.2',6262,24,40,R=True)
    udp7 = Iperf3Out(**udp7)

    #####
    udp3Mbps = [inter['sum']['bits_per_second']/(10**6) for inter in udp3.intervals]
    udp12Mbps = [inter['sum']['bits_per_second']/(10**6) for inter in udp12.intervals]
    udp7Mbps = [inter['sum']['bits_per_second']/(10**6) for inter in udp7.intervals]
    udp14Mbps = [inter['sum']['bits_per_second']/(10**6) for inter in udp14.intervals]

    udparray = np.array([udp3Mbps,udp7Mbps,udp12Mbps,udp14Mbps]).T

    tensor = tf.cast(udparray, tf.float32)
    return tensor


def A2G(a):
    W = np.where(a-np.eye(a.shape[1])==1)
    W = list(zip(W[0].tolist(),W[1].tolist()))
    g = nx.Graph(W)
    return g


def random_uncycle_graph(num_v):
    g = nx.generators.random_graphs.dense_gnm_random_graph(num_v,num_v*2)
    while True:
        try:
            edges = nx.cycles.find_cycle(g)
        except:
            break
        g.remove_edge(*edges[0])
    A = nx.adjacency_matrix(g).toarray()
    A = A + np.eye(A.shape[0])
    Atld = localpooling_filter(A)
    X = np.eye(A.shape[1])
    X = X.reshape(X.shape[0],1)
    return A, Atld, X

def gen_batch_graphs():
    batch = 500
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(5, 8,batch)
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
    rgi = np.random.randint(0,w_sum)
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
    A = A + np.eye(A.shape[0])
    Atld = localpooling_filter(A)
    X = np.eye(A.shape[0])
    # X = X.reshape(X.shape[0],X.shape[1],1)
    return A, Atld, X, rgi

def gen_nx_graphs(batch=100, want=[1,1,1,1,1,1,1,1],minN=6,maxN=10):
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        C_list = []
        gen_v = np.random.randint(minN, maxN,batch)
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
            bottom_pad = np.zeros((maxN- x.shape[0], maxN))

            x = np.concatenate([x, left_pad],axis=1)
            x = np.concatenate([x, bottom_pad],axis=0)
            X_list[i] = x
        
        


        X, A = numpy_to_batch(X_list, A_list)
        _, Atld = numpy_to_batch(X_list, Atld_list)
        A = np.array(A)
        Atld = np.array(Atld)
        yield X, A, Atld, C_list

def gen_nx_random_uncycle_graphs():
    batch = 1000
    while True:
        Atld_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(5, 10,batch)
        for num_v in gen_v:
            r_choice = np.random.randint(2)
            if r_choice == 0:
                A, Atld, X = nx_graph(num_v)
            else:
                A, Atld, X = random_uncycle_graph(num_v)
            A_list.append(A)
            Atld_list.append(Atld)
            X_list.append(X)

        X, A = numpy_to_batch(X_list, A_list)
        _, Atld = numpy_to_batch(X_list, Atld_list)
        A = A.toarray()
        Atld = Atld.toarray()
        yield A, Atld, X


def list2graph(nodelist, maxN=None):
    A_list = []
    Atld_list = []
    X_list = []

    for elements in nodelist:
        g = Graph(elements)
        a = adjacency_matrix(g,sorted(g.nodes())).toarray()
        i = np.eye(a.shape[0])
        a = a + i
        a_tld = localpooling_filter(a)
        x = np.eye(a.shape[0])
        A_list.append(a)
        Atld_list.append(a_tld)
        X_list.append(x)
    
    X, A = numpy_to_batch(X_list, A_list)
    _, Atld = numpy_to_batch(X_list, Atld_list)

    X = tf.cast(X,tf.float32)
    A = tf.cast(A,tf.float32)
    Atld = tf.cast(Atld,tf.float32)

    if maxN is None:
        return X, A, Atld
    elif maxN <= X.shape[1]:
        X = X[:, :maxN, :maxN]
        A = A[:, :maxN, :maxN]
        Atld = Atld[:, :maxN, :maxN]
        return X, A, Atld
    else:
        left_pad = tf.zeros((X.shape[0], X.shape[1], maxN - X.shape[2]))
        bottom_pad = tf.zeros((X.shape[0], maxN- X.shape[1], maxN))

        X = tf.concat([X, left_pad],axis=2)
        X = tf.concat([X, bottom_pad],axis=1)

        A = tf.concat([A,left_pad],axis=2)
        A = tf.concat([A, bottom_pad],axis=1)

        Atld = tf.concat([Atld,left_pad],axis=2)
        Atld = tf.concat([Atld, bottom_pad],axis=1)
        return X, A, Atld


def graphs_stats(Adjs):
    maxD=0
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
            tree = nx.algorithms.traversal.breadth_first_search.bfs_tree(g,n)
            depths = nx.shortest_path_length(tree,n).values()        
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


def history_figure(history, figsize=(16,8), legend_fontsize=18, axes_label_fondsize=22,ticks_fontsize=16,markersize=15, save_obj=None):
    max_harm = np.max(history["harmonic_score"])
    max_harm_arg = np.argmax(history["harmonic_score"])

    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.plot(history["train_cost"],label=r"$train \, cost \, (seen)$")
    plt.plot(history["val_cost"],label=r"$val \, cost \, (seen)$")
    plt.plot(history["test_cost"],label=r"$test \, cost \, (unseen)$")
    plt.xlabel(r"$epochs \, \#$", fontsize=axes_label_fondsize)
    plt.ylabel(r"$Cost$", fontsize=axes_label_fondsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    plt.subplot(1,2,2)
    plt.plot(history["train_score"],label=r"$train \, score \, (seen)$")
    plt.plot(history["val_score"],label=r"$val \, score \, (seen)$")
    plt.plot(history["test_score"],label=r"$test \, score \, (unseen)$")
    plt.plot(history["harmonic_score"],label=r"$harmonic \, score$")
    plt.plot(max_harm_arg,max_harm,'*',label=r'$harmonic \, score \, (best)$', markersize=15)
    plt.xlabel(r"$epochs \, \#$", fontsize=axes_label_fondsize)
    plt.ylabel(r"$Accuracy \, Score$", fontsize=axes_label_fondsize)
    plt.legend(fontsize=legend_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    print('Harmonic Score (Best): {}'.format(max_harm))
    print('Val Score (Hs Best):', history["val_score"][max_harm_arg])
    print('Test Score (Hs Best):', history["test_score"][max_harm_arg])
    if save_obj is not None:
        plt.savefig('./{}/DataFigures/{}/{}-{}.eps'.format(*save_obj), format='eps')