from spektral.utils import localpooling_filter
import tensorflow as tf
import numpy as np
from .models import SGAE, miniGAEv1, ExtendedNN, L2CategoricalCrossentropy, L2MeanSquaredError
from .datasets import GaeDataset, ZeroShotDataset, ENNdataset
from sklearn.neighbors import KNeighborsClassifier
from .methods import history_figure, pca_denoising_figure, pca_denoising_preprocessing, print_confmtx
import json
from matplotlib import pyplot as plt


def get_results(dz):
    # this is simple for 1 loop
    def forloop(dzds):
        for x, e_emb, r, a in dzds:
            pass
        return x, e_emb, r, a
    
    x_train, r_emb_train, r_train, a_train = forloop(dz.train)
    x_val, r_emb_val, r_val, a_val = forloop(dz.val)
    x_test, r_emb_test, r_test, a_test = forloop(dz.test)

    return (x_train, r_emb_train, r_train, a_train), (x_val, r_emb_val, r_val, a_val), (x_test, r_emb_test, r_test, a_test)


def main():
    embs_id = '32'
    embsD = 32

    labels = ['0','1','2']
    # all combinations seen (s), unseen (u):
    # {s: ['0','1'], u:['2']}, {s: ['0','2'], u:['1']}, {s: ['1','2'], u:['0']}
    seen_labels = ['0','2']
    unseen_labels = ['1']
    outfilekey = 's-{}-u-{}'.format('-'.join(seen_labels), '-'.join(unseen_labels))

    extd = ENNdataset('32', '.')

    knn = KNeighborsClassifier(1)
    r_emb = extd.get_r_embs_dict()
    X = tf.concat([r_emb[label].reshape(1,-1) for label in labels],axis=0)
    Xall = tf.concat([r_emb[key].reshape(1,-1) for key in r_emb.keys()],axis=0)
    labels_all = list(r_emb.keys())
    Y = extd.get_r_onehot_from_labels([label for label in labels])
    knn.fit(X,Y)
    #################################################################
    
    A = np.array([[
    [1,1,1,1,1],
    [1,1,0,0,0],
    [1,0,1,0,0],
    [1,0,0,1,0],
    [1,0,0,0,1]]]).astype(np.float32)
    A = tf.cast(A, tf.float32)
    w_p = float(A.shape[1]*A.shape[2] * A.shape[2] - tf.reduce_sum(A)) / tf.reduce_sum(A)


    
    es = {'es_strategy':'patience', 'es_patience':10, 'es_min_delta':0.1, 'es_metric':'harmonic_score'}
    es = {'es_strategy':'first_drop', 'es_metric':'harmonic_score'}
    
    # train
    # hyper parameters by bayes opt
    l = 98.81297973768632
    epochs = 108

    enn = ExtendedNN(embsD,[5,64], w_p=w_p, knn=knn,learning_rate=1e-4, lamda=l)
    enn.train(extd.train, epochs, extd.val, extd.test, history_learning_process=False)
    l2mse = L2MeanSquaredError()
    l2cce = L2CategoricalCrossentropy()
    enn.set_score_mode(False)

    for batch in extd.val:
        out = enn.sgae.sgcn0(batch[0], batch[1])
        print(enn.significance(out))
    
    # change this to True
    h = enn.history
    h['train_cost'] = [float(v) for v in h['train_cost']]
    h['train_score'] = [float(v) for v in h['train_score']]
    h['val_cost'] = [float(v) for v in h['val_cost']]
    h['val_score'] = [float(v) for v in h['val_score']]
    h['test_cost'] = [float(v) for v in h['test_cost']]
    h['test_score'] = [float(v) for v in h['test_score']]
    h['harmonic_score'] = [float(v) for v in h['harmonic_score']]
    
    save_history_to_file = False
    if save_history_to_file:
        with open('./data/extnn/history-{}.json'.format(outfilekey), 'w') as outfile:
                json.dump(h, outfile, indent=4)
    
    save_obj = ['.','notebooks','extnn', 'cost-score', outfilekey]
    # change this to history and save_obj
    # history_figure(h, save_obj=None)
    enn.set_score_mode(True)
    print('val', enn.score_mtr.metric_dataset(extd.val))
    print('test', enn.score_mtr.metric_dataset(extd.test))
    print('HM', enn.score_mtr.metric_dataset(extd.test)*2*enn.score_mtr.metric_dataset(extd.val)/(enn.score_mtr.metric_dataset(extd.val)+enn.score_mtr.metric_dataset(extd.test)))
    enn.set_score_mode(False)

    print_confmtx(enn, extd, outfilekey)
    (A_train, _, _, p_train), _, _ = enn.get_results(extd, True)
    print(A_train[0], tf.cast(p_train[0][0]>0.5,tf.int8), p_train[0][0])

    pca_vl ,pca_ts, pca_emb, knn_pca = pca_denoising_preprocessing(enn, extd, Xall, Y)
    save_obj = ['.','notebooks','extnn', 'node-emb', outfilekey]
    pca_denoising_figure(pca_vl ,pca_ts, pca_emb, knn_pca, labels_all, save_obj=None)

main()