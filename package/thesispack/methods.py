import os
import subprocess
import json
import numpy as np
import tensorflow as tf
import time
from spektral.utils import localpooling_filter
from spektral.utils.data import numpy_to_batch, numpy_to_disjoint
from networkx import Graph, adjacency_matrix
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib as mlb

# ---------------------- #
# routing -------------- #
# ---------------------- #
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


# ---------------------- #
# graphs --------------- #
# ---------------------- #
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


# ---------------------- #
# neural networks ------ #
# ---------------------- #
def history_figure(history, figsize=(16,8), legend_fontsize=18, axes_label_fondsize=22,ticks_fontsize=16,markersize=15, save_obj=None):

    zstype = lambda ksplit0: '(seen)' if ksplit0 in ['train', 'val'] else '(unseen)'
    def score_cost_plot(plot_type='cost'):
        for k, hasv in hasattr_list:
            ksplit = k.split('_')
            if hasv and ksplit[1] == plot_type:
                zs = zstype(ksplit[0])
                klabel = ' \, '.join(ksplit)
                label = '{} \, {}'.format(klabel, zs)
                plt.plot(history[k],label=r'${}$'.format(label))

        if plot_type == 'score':
            try:
                plt.plot(history["harmonic_score"],label=r"$harmonic \, score$")
                max_harm = np.max(history["harmonic_score"])
                max_harm_arg = np.argmax(history["harmonic_score"])
                plt.plot(max_harm_arg,max_harm,'*',label=r'$harmonic \, score \, (best)$', markersize=15, color='tab:purple')
                print('Harmonic Score (Best): {}'.format(max_harm))
                print('Val Score (Hs Best):', history["val_score"][max_harm_arg])
                print('Test Score (Hs Best):', history["test_score"][max_harm_arg])
            except:
                pass


        plt.xlabel(r"$\# \, of \, epochs$", fontsize=axes_label_fondsize)
        plt.ylabel(r'${}$'.format(plot_type), fontsize=axes_label_fondsize)
        plt.legend(fontsize=legend_fontsize)
        plt.xticks(fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

    hasattr_list = [(k, any(v)) for k, v in history.items() if k != "harmonic_score"]
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    score_cost_plot('cost')
    plt.subplot(1,2,2)
    score_cost_plot('score')


    if save_obj is not None:
        plt.savefig('{}/{}/DataFigures/{}/{}-{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


# only for ENN fix for all maybe use self.__status @property
def print_confmtx(model, dataset, lerninfo:str, indexes_list:list):
    """
    docstring
    """
    def confmtx(y, yhat):
        """
        docstring
        """
        confmtx = np.zeros((y.shape[1], y.shape[1]))  
        y = np.argmax(y,axis=1)
        yhat = np.argmax(yhat,axis=1)
        y_concat = np.concatenate([y.reshape(-1,1),yhat.reshape(-1,1)],axis=1)

        for y1, y2 in y_concat:
            confmtx[y1,y2]+=1

        return confmtx
    
    
    (_, outs_tr, p_train), (_, outs_vl, p_val), (_, outs_ts, p_test) = model.get_results(dataset, True)
    
    indxr = indexes_list[0]
    indxa = indexes_list[1]

    (r_train, a_train) = outs_tr[indxr], outs_tr[indxa]
    (r_val, a_val) = outs_vl[indxr], outs_vl[indxa]
    (r_test, a_test) = outs_ts[indxr], outs_ts[indxa]


    lr = [[r_train, p_train[indxr]], [r_val, p_val[indxr]], [ r_test, p_test[indxr]]]
    la = [[a_train, p_train[indxa]], [a_val, p_val[indxa]], [ a_test, p_test[indxa]]]


    print('a', lerninfo)
    for expl, (tr, pr) in zip(['train', 'val', 'test'], la):
        print(expl)
        print(confmtx(tr, pr))

    print('r', lerninfo) 
    for expl, (tr, pr) in zip(['train', 'val', 'test'], lr):
        print(expl)
        print(confmtx(tr, pr))


def sift_point_to_best(best_point, point, sift_dist):
    dist = np.sqrt(np.sum((point-best_point)**2))
    a = sift_dist/dist
    new_point = np.array([
        point[0]*a+(1-a)*best_point[0],
        point[1]*a+(1-a)*best_point[1]
        ])
    return new_point[0], new_point[1]


def pca_denoising(p_expls, pca_emb, pca_expls, knn, knn_pca, Dx=0, Dy=1):
    
    diferent_cls_ts = np.where(knn_pca.kneighbors(pca_expls[:,[Dx, Dy]])[1] != knn.kneighbors(p_expls)[1])[0]
    knn_n = knn.kneighbors(p_expls)[1]
    for ii in diferent_cls_ts:
        sift_dist = knn_pca.kneighbors(pca_expls[:,[Dx, Dy]])[0][ii][0]
        
        x, y = sift_point_to_best(pca_emb[knn_n[ii][0],[Dx, Dy]], pca_expls[ii,[Dx, Dy]], sift_dist)
        pca_expls[ii, Dx] = x
        pca_expls[ii, Dy] = y

    return pca_expls

# Z is true Embeddings
def pca_denoising_preprocessing(model, dataset, Z, Y, embidx=0):
    (_, _, p_train), (_, _, p_val), (_, _, p_test) = model.get_results(dataset, False)

    pca = PCA(n_components=2)
    pca.fit(Z)
    
    pca_emb = pca.transform(Z)
    pca_vl = pca.transform(p_val[embidx])
    pca_ts = pca.transform(p_test[embidx])

    Dx = 0
    Dy = 1
    # :3 means up to room 2, change it to be more general
    knn_pca = KNeighborsClassifier(1)
    knn_pca.fit(pca_emb[:3,[Dx, Dy]], Y)


    pca_vl = pca_denoising(p_val[embidx], pca_emb, pca_vl, model.knn, knn_pca)
    pca_ts = pca_denoising(p_test[embidx], pca_emb, pca_ts, model.knn, knn_pca)

    return pca_vl ,pca_ts, pca_emb, knn_pca

def pca_denoising_figure(pca_vl ,pca_ts, pca_emb, knn_pca, Zlabels, save_obj=None):
    mlb.style.use('default')
    dpi = 100
    xmin = np.min(pca_emb[:,0])
    xmin += np.sign(xmin)
    xmax = np.max(pca_emb[:,0])
    xmax += np.sign(xmax)
    ymin = np.min(pca_emb[:,1])
    ymin += np.sign(ymin)
    ymax = np.max(pca_emb[:,1])
    ymax += np.sign(ymax)

    xlin = np.linspace(xmin,xmax,dpi)
    ylin = np.linspace(ymin,ymax,dpi)
    xx, yy = np.meshgrid(ylin, ylin)
    knn_space = np.argmax(knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]),axis=1)
    knn_space = knn_space.reshape(xx.shape)

    Dx = 0
    Dy = 1

    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(pca_emb[:3,Dx],pca_emb[:3,Dy],'o',label='embs', markersize=15)
    plt.plot(pca_vl[:,Dx],pca_vl[:,Dy],'*',label='val', markersize=10)
    plt.plot(pca_ts[:,Dx],pca_ts[:,Dy],'*',label='test', markersize=10)

    for (v,l) in zip(pca_emb, Zlabels[:3]):
        plt.text(v[Dx],v[Dy],l, fontsize=20)

    ax.contourf(xx, yy, knn_space, cmap=plt.get_cmap('tab20c'), levels=2)
    plt.legend()
    if save_obj is not None:
        plt.savefig('{}/{}/DataFigures/{}/{}-{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


def n_identity_matrix(N):
    return tf.cast([[[1 if i==j and j==w  else 0 for i in range(N)] for j in range(N)] for w in range(N)], tf.float32)


