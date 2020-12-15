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

def udp_figure(udpresults_dict,figsize=(16,8),ylim_max=23,legend_fontsize=16,legend_loc='upper left', 
    axes_label_fondsize=20,ticks_fontsize=16,plot_icons=['*-','o-','^-'],markersize=10,save_obj=None):
    plt.figure(figsize=figsize)
    for (key, value), plot_icon  in zip(udpresults_dict.items(), plot_icons):
        udptime = [t for t in range(1, len(value)+1)]
        plt.plot(udptime, value ,plot_icon,markersize=markersize,label=key)

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.xlabel(r'$time \, (s)$',fontsize=axes_label_fondsize)
    plt.ylabel(r'$UDP \, Throughput \, (Mbps)$', fontsize=axes_label_fondsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    
    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0,ylim_max)
    
    if save_obj is not None:
        plt.savefig('{}/eps_pics/{}/{}.eps'.format(save_obj[0], save_obj[1], save_obj[2]), format='eps')


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
    A_hat = localpooling_filter(A)
    X = np.eye(A.shape[1])
    X = X.reshape(X.shape[0],1)
    return A, A_hat, X

def gen_batch_graphs():
    batch = 500
    while True:
        A_hat_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(5, 8,batch)
        for num_v in gen_v:
            A, A_hat, X = random_uncycle_graph(num_v)
            A_list.append(A)
            A_hat_list.append(A_hat)
            X_list.append(X)
        X, A, I = numpy_to_disjoint(X_list, A_list)
        _, A_hat, _ = numpy_to_disjoint(X_list, A_hat_list)
        A = A.toarray()
        A_hat = A_hat.toarray()
        yield A, A_hat, X


def nx_graph(num_v, want):
    w_sum = sum(want)
    want = [sum(want[:i]) if want[i] == 1 else -1 for i in range(len(want))]
    r = np.random.randint(0,w_sum)
    if r == want[0]:
        g = nx.cycle_graph(num_v)
    elif r == want[1]:
        g = nx.star_graph(num_v - 1)
    elif r == want[2]:
        g = nx.wheel_graph(num_v)
    elif r == want[3]:
        g = nx.complete_graph(num_v)
    elif r == want[4]:
        path_len = np.random.randint(2, num_v // 2)
        g = nx.lollipop_graph(m=num_v - path_len, n=path_len)
    elif r == want[5]:
        g = nx.hypercube_graph(np.log2(num_v).astype('int'))
        g = nx.convert_node_labels_to_integers(g)
    elif r == want[6]:
        g = nx.circular_ladder_graph(num_v // 2)
    elif r == want[7]:
        n_rows = np.random.randint(2, num_v // 2)
        n_cols = num_v // n_rows
        g = nx.grid_graph([n_rows, n_cols])
        g = nx.convert_node_labels_to_integers(g)
    
    A = nx.adjacency_matrix(g)
    A = A + np.eye(A.shape[0])
    A_hat = localpooling_filter(A)
    X = np.eye(A.shape[0])
    # X = X.reshape(X.shape[0],X.shape[1],1)
    return A, A_hat, X, r

def gen_nx_graphs(batch=100, want=[1,1,1,1,1,1,1,1],minN=6,maxN=10):
    while True:
        A_hat_list = []
        A_list = []
        X_list = []
        C_list = []
        gen_v = np.random.randint(minN, maxN,batch)
        maxN = 0
        for num_v in gen_v:
            A, A_hat, X, C = nx_graph(num_v, want)
            if maxN < A.shape[0]:
                maxN = A.shape[0]
            A_list.append(A)

            A_hat_list.append(A_hat)
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
        _, A_hat = numpy_to_batch(X_list, A_hat_list)
        A = np.array(A)
        A_hat = np.array(A_hat)
        yield X, A, A_hat, C_list

def gen_nx_random_uncycle_graphs():
    batch = 1000
    while True:
        A_hat_list = []
        A_list = []
        X_list = []
        gen_v = np.random.randint(5, 10,batch)
        for num_v in gen_v:
            r_choice = np.random.randint(2)
            if r_choice == 0:
                A, A_hat, X = nx_graph(num_v)
            else:
                A, A_hat, X = random_uncycle_graph(num_v)
            A_list.append(A)
            A_hat_list.append(A_hat)
            X_list.append(X)

        X, A = numpy_to_batch(X_list, A_list)
        _, A_hat = numpy_to_batch(X_list, A_hat_list)
        A = A.toarray()
        A_hat = A_hat.toarray()
        yield A, A_hat, X


def list2graph(nodelist, maxN=None):
    A_list = []
    A_hat_list = []
    X_list = []

    for elements in nodelist:
        g = Graph(elements)
        a = adjacency_matrix(g,sorted(g.nodes())).toarray()
        i = np.eye(a.shape[0])
        a = a + i
        a_hat = localpooling_filter(a)
        x = np.eye(a.shape[0])
        A_list.append(a)
        A_hat_list.append(a_hat)
        X_list.append(x)
    
    X, A = numpy_to_batch(X_list, A_list)
    _, A_hat = numpy_to_batch(X_list, A_hat_list)

    X = tf.cast(X,tf.float32)
    A = tf.cast(A,tf.float32)
    A_hat = tf.cast(A_hat,tf.float32)

    if maxN is None:
        return X, A, A_hat
    elif maxN <= X.shape[1]:
        X = X[:, :maxN, :maxN]
        A = A[:, :maxN, :maxN]
        A_hat = A_hat[:, :maxN, :maxN]
        return X, A, A_hat
    else:
        left_pad = tf.zeros((X.shape[0], X.shape[1], maxN - X.shape[2]))
        bottom_pad = tf.zeros((X.shape[0], maxN- X.shape[1], maxN))

        X = tf.concat([X, left_pad],axis=2)
        X = tf.concat([X, bottom_pad],axis=1)

        A = tf.concat([A,left_pad],axis=2)
        A = tf.concat([A, bottom_pad],axis=1)

        A_hat = tf.concat([A_hat,left_pad],axis=2)
        A_hat = tf.concat([A_hat, bottom_pad],axis=1)
        return X, A, A_hat


class Iperf3Out:
    def __init__(self, **entries):
        self.__dict__.update(entries)
