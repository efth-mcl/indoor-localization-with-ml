from spektral.utils import localpooling_filter
import tensorflow as tf
import numpy as np
from .models import ExtendedNN, SGAEsolo
from .datasets import ZeroShotDataset, ENNdataset, SGAE_dataset
from sklearn.neighbors import KNeighborsClassifier
from .methods import history_figure, pca_denoising_figure, pca_denoising_preprocessing, print_confmtx, n_identity_matrix
import json
from matplotlib import pyplot as plt
from thesispack.methods import list2graph
from bayes_opt import BayesianOptimization

def Pr0(Atld, X):
    I0 = n_identity_matrix(Atld.shape[0])
    I2 = n_identity_matrix(Atld.shape[2])
    Xpr = tf.tensordot(I0, Atld, [[1],[0]])
    Xpr = tf.tensordot(Xpr, I2, [[3],[1]])
    Xpr = tf.tensordot(Xpr, X, [[1,4],[0,1]])
    return Xpr


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
    knn.fit(X, Y)
    #################################################################

    X, A, Atld = list2graph(SGAE_dataset['route_links'])

    train = tf.data.Dataset.from_tensor_slices((
        Pr0(Atld, X) , Atld, A
    )).batch(128)

    w_p = float(A.shape[1]*A.shape[2] * A.shape[2] - tf.reduce_sum(A)) / tf.reduce_sum(A)
    norm = A.shape[1] * A.shape[2] / float((A.shape[1] * A.shape[2] - tf.reduce_sum(A)) * 2)

    # train solo SGAE
    solo = SGAEsolo([A.shape[1],64],w_p, norm,learning_rate=1e-1)
    solo.train(train,10,history_learning_process=True)
    history_figure(solo.history, save_obj=None)

    # train ExtendedNN
    enn = ExtendedNN(embsD,[A.shape[1],64], w_p=w_p, norm=norm, knn=knn,learning_rate=1e-1, early_stop_vars=None)
    enn.train(extd.train, 10, extd.val, extd.test, history_learning_process=True)
    enn.set_score_mode(False)
    for batch in extd.val:
        out = enn.sgae.sgcn0(batch[0], batch[1])
        print(enn.significance(out))
    # change this to True
    h = enn.history
    
    save_history_to_file = False
    if save_history_to_file:
        with open('./data/extnn/history-{}.json'.format(outfilekey), 'w') as outfile:
                json.dump(h, outfile, indent=4)
    
    save_obj = ['.','notebooks','extnn', 'cost-score', outfilekey]
    # change this to history and save_obj
    history_figure(enn.history, save_obj=None)
    enn.set_score_mode(True)
    print('val', enn.score_mtr.metric_dataset(extd.val))
    print('test', enn.score_mtr.metric_dataset(extd.test))
    print('HM', enn.score_mtr.metric_dataset(extd.test)*2*enn.score_mtr.metric_dataset(extd.val)/(enn.score_mtr.metric_dataset(extd.val)+enn.score_mtr.metric_dataset(extd.test)))
    enn.set_score_mode(False)

    print_confmtx(enn, extd, outfilekey, [1,2])
    (_, outs_train, p_train), _, _ = enn.get_results(extd, True)
    print(outs_train[0][0], tf.cast(p_train[0][0]>0.5,tf.int8), p_train[0][0])

    pca_vl, pca_ts, pca_emb, knn_pca = pca_denoising_preprocessing(enn, extd, Xall, Y, 1)
    save_obj = ['.','notebooks','extnn', 'node-emb', outfilekey]
    pca_denoising_figure(pca_vl ,pca_ts, pca_emb, knn_pca, labels_all, save_obj=None)

main()