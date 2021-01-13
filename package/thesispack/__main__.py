from spektral.utils import localpooling_filter
import tensorflow as tf
import numpy as np
from .models import Pr0, SGAE, miniGAEv1, ExtendedNN, L2CategoricalCrossentropy, L2MeanSquaredError
from .datasets import GaeDataset, TEST_PATHS, TRAIN_PATHS, ZeroShotDataset
from sklearn.neighbors import KNeighborsClassifier
from .methods import history_figure
import json


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
    test = tf.test.TestCase(methodName='runTest')

    embs_id = '32'
    embsD = 32

    dz = ZeroShotDataset(embs_id)
    labels = ['0','1','2']
    # all combinations seen (s), unseen (u):
    # {s: ['0','1'], u:['2']}, {s: ['0','2'], u:['1']}, {s: ['1','2'], u:['0']}
    seen_labels = ['0','2']
    unseen_labels = ['1']
    outfilekey = 's-{}-u-{}'.format('-'.join(seen_labels), '-'.join(unseen_labels))

    dz.load_data(TRAIN_PATHS, TEST_PATHS,seen_labels,unseen_labels)

    knn = KNeighborsClassifier(1)
    r_emb = dz.get_r_embs_dict()
    X = tf.concat([r_emb[label].reshape(1,-1) for label in labels],axis=0)
    Xall = tf.concat([r_emb[key].reshape(1,-1) for key in r_emb.keys()],axis=0)
    labels_all = list(r_emb.keys())
    Y = dz.get_r_onehot_from_labels([label for label in labels])
    knn.fit(X,Y)
    #################################################################
    


    A = np.array([[
    [1,1,1,1,1],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]]]).astype(np.float32)
    X = np.eye(A.shape[1]).reshape(1,A.shape[1],A.shape[1])
    Atld = localpooling_filter(A)
    A = tf.cast(A, tf.float32)
    X = tf.cast(X, tf.float32)
    Atld = tf.cast(Atld, tf.float32)
    Xpr = Pr0(Atld, X)
    w_p = float(A.shape[1]*A.shape[2] * A.shape[2] - tf.reduce_sum(A)) / tf.reduce_sum(A)

    (x_train, r_emb_train, r_train, a_train), (x_val, r_emb_val, r_val, a_val), (x_test, r_emb_test, r_test, a_test) = get_results(dz)

    Ntrian = x_train.shape[0]
    dext_train = tf.data.Dataset.from_tensor_slices((
            tf.concat(Ntrian*[Xpr],axis=0), tf.concat(Ntrian*[Atld],axis=0), x_train, tf.concat(Ntrian*[A],axis=0), r_emb_train, r_train, a_train
        )).batch(128)
    
    Nval = x_val.shape[0]
    dext_val = tf.data.Dataset.from_tensor_slices((
            tf.concat(Nval*[Xpr],axis=0), tf.concat(Nval*[Atld],axis=0), x_val, tf.concat(Nval*[A],axis=0), r_emb_val, r_val, a_val
        )).batch(128)

    Ntest = x_test.shape[0]
    dext_test = tf.data.Dataset.from_tensor_slices((
            tf.concat(Ntest*[Xpr],axis=0), tf.concat(Ntest*[Atld],axis=0), x_test, tf.concat(Ntest*[A],axis=0), r_emb_test, r_test, a_test
        )).batch(128)
    
    es = {'es_strategy':'patience', 'es_patience':10, 'es_min_delta':0.1, 'es_metric':'harmonic_score'}
    enn = ExtendedNN(embsD,[X.shape[-1],64], w_p=w_p, knn=knn,learning_rate=5e-3, early_stop_vars=None)

    # train
    enn.train(dext_train, 500, dext_val, dext_test, history_learning_process=True)
    l2mse = L2MeanSquaredError()
    l2cce = L2CategoricalCrossentropy()
    enn.set_score_mode(False)
    ls = []
    for batch in dext_val:
        out = enn(batch[:3])
        ls.append(l2mse(out[1],batch[4])+l2cce(out[2],batch[6]))
    
    print(np.argmin(ls),ls)
    sgcnouts_val = enn.sgae(tf.concat(Nval*[Xpr],axis=0), tf.concat(Nval*[Atld],axis=0))
    s = []
    for out in  sgcnouts_val[:-1]:
        s.append(enn.significance(out))
    
    print(s[np.argmin(ls)][0])
    # change this to True
    save_history_to_file = True
    h = enn.history
    h['train_cost'] = [float(v) for v in h['train_cost']]
    h['train_score'] = [float(v) for v in h['train_score']]
    h['val_cost'] = [float(v) for v in h['val_cost']]
    h['val_score'] = [float(v) for v in h['val_score']]
    h['test_cost'] = [float(v) for v in h['test_cost']]
    h['test_score'] = [float(v) for v in h['test_score']]
    h['harmonic_score'] = [float(v) for v in h['harmonic_score']]
    if save_history_to_file:
        with open('./data/extnn/history-{}.json'.format(outfilekey), 'w') as outfile:
                json.dump(h, outfile, indent=4)
    else:
        with open('./data/extnn/history-{}.json'.format(outfilekey), 'r') as f:
            history = json.load(f)
    
    save_obj = ['notebooks','extnn', 'cost-score', outfilekey]
    # change this to history and save_obj
    history_figure(h, save_obj=save_obj)
main()