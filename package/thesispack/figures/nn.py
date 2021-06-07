import numpy as np
import matplotlib as mlb
import tensorflow as tf
import matplotlib.pyplot as plt
from thesispack.datasets import WIFIRTTDataset


def history_figure(history, figsize=(16, 8), legend_fontsize=18, axes_label_fondsize=22, ticks_fontsize=16,
                   markersize=15, save_obj=None):
    zstype = lambda ksplit0: '(seen)' if ksplit0 in ['train', 'val'] else '(unseen)'

    def score_cost_plot(plot_type='cost'):
        for k, hasv in hasattr_list:
            ksplit = k.split('_')
            if hasv and ksplit[1] == plot_type:
                zs = zstype(ksplit[0])
                klabel = ' \, '.join(ksplit)
                label = '{} \, {}'.format(klabel, zs)
                plt.plot(history[k], label=r'${}$'.format(label))

        if plot_type == 'score':
            try:
                max_harm = np.max(history["harmonic_score"])
                max_harm_arg = np.argmax(history["harmonic_score"])
                plt.plot(max_harm_arg, max_harm, '*', label=r'$harmonic \, score \, (best)$', markersize=15,
                         color='tab:purple')
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

    hasattr_list = [(k, any(v)) for k, v in history.items()]
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    score_cost_plot('cost')
    plt.subplot(1, 2, 2)
    score_cost_plot('score')

    if save_obj is not None:
        plt.savefig('{}/{}/DataFigures/{}/{}-{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


# only for ENN fix for all maybe use self.__status @property
def print_confmtx(model, dataset, lerninfo: str, indexes_list: list):
    """
    docstring
    """

    def confmtx(y, yhat):
        """
        docstring
        """
        confmtx = np.zeros((y.shape[1], y.shape[1]))
        y = np.argmax(y, axis=1)
        yhat = np.argmax(yhat, axis=1)
        y_concat = np.concatenate([y.reshape(-1, 1), yhat.reshape(-1, 1)], axis=1)

        for y1, y2 in y_concat:
            confmtx[y1, y2] += 1

        return confmtx

    (_, outs_tr, p_train), (_, outs_vl, p_val), (_, outs_ts, p_test) = model.get_results(dataset, True)

    indxr = indexes_list[0]
    indxa = indexes_list[1]

    (r_train, a_train) = outs_tr[indxr], outs_tr[indxa]
    (r_val, a_val) = outs_vl[indxr], outs_vl[indxa]
    (r_test, a_test) = outs_ts[indxr], outs_ts[indxa]

    lr = [[r_train, p_train[indxr]], [r_val, p_val[indxr]], [r_test, p_test[indxr]]]
    la = [[a_train, p_train[indxa]], [a_val, p_val[indxa]], [a_test, p_test[indxa]]]

    print('a', lerninfo)
    for expl, (tr, pr) in zip(['train', 'val', 'test'], la):
        print(expl)
        print(confmtx(tr, pr))

    print('r', lerninfo)
    for expl, (tr, pr) in zip(['train', 'val', 'test'], lr):
        print(expl)
        print(confmtx(tr, pr))


def pca_denoising_figure(pca_vl, pca_ts, pca_emb, knn_pca, Zlabels, save_obj=None):
    mlb.style.use('default')
    dpi = 100
    xmin = np.min(pca_emb[:, 0])
    xmin += np.sign(xmin)
    xmax = np.max(pca_emb[:, 0])
    xmax += np.sign(xmax)
    ymin = np.min(pca_emb[:, 1])
    ymin += np.sign(ymin)
    ymax = np.max(pca_emb[:, 1])
    ymax += np.sign(ymax)

    xlin = np.linspace(xmin, xmax, dpi)
    ylin = np.linspace(ymin, ymax, dpi)
    xx, yy = np.meshgrid(ylin, ylin)
    knn_space = np.argmax(knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    knn_space = knn_space.reshape(xx.shape)

    Dx = 0
    Dy = 1

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(pca_emb[:3, Dx], pca_emb[:3, Dy], 'o', label='embs', markersize=15)
    plt.plot(pca_vl[:, Dx], pca_vl[:, Dy], '*', label='val', markersize=10)
    plt.plot(pca_ts[:, Dx], pca_ts[:, Dy], '*', label='test', markersize=10)

    for (v, l) in zip(pca_emb, Zlabels[:3]):
        plt.text(v[Dx], v[Dy], l, fontsize=20)

    ax.contourf(xx, yy, knn_space, cmap=plt.get_cmap('tab20c'), levels=2)
    plt.legend()
    if save_obj is not None:
        plt.savefig('{}/{}/DataFigures/{}/{}-{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


def plot_psxy(model, dataset: WIFIRTTDataset, plotd='train', maxdist=np.inf, save_obj=None):
    pxy_t = []
    pxy_t_ = []
    if plotd == 'train':
        for t in dataset.train:
            pxy_t.append(t[-1])
            pxy_t_.append(model(t)[0])
    elif plotd == 'val':
        for t in dataset.val:
            pxy_t.append(t[-1])
            pxy_t_.append(model(t)[0])
    elif plotd == 'test':
        for t in dataset.test:
            pxy_t.append(t[-1])
            pxy_t_.append(model(t)[0])

    pxy_t_ = tf.concat(pxy_t_, axis=0)
    pxy_t = tf.concat(pxy_t, axis=0)

    max_xy, min_xy = dataset.get_min_max_pos()
    pxy_t = pxy_t * (max_xy - min_xy) + min_xy
    pxy_t_ = pxy_t_ * (max_xy - min_xy) + min_xy

    dists = tf.sqrt(tf.reduce_sum((pxy_t - pxy_t_) ** 2, axis=1)).numpy()
    dists = dists[np.where(dists <= maxdist)]
    plt.figure(figsize=(16, 8))
    hist_cumul = plt.hist(dists, density=True, histtype='step', cumulative=True, bins=150)
    p90 = np.where(hist_cumul[0] >= 0.9)[0][0]
    print(hist_cumul[1][p90])
    plt.scatter(hist_cumul[1][p90], hist_cumul[0][p90])
    plt.figure(figsize=(16, 8))
    plt.hist(dists, bins=150)
    plt.figure(figsize=(16, 8))


def plot_psxy2(model, dataset: WIFIRTTDataset, plotd='train', kfirst=10, plot_arrows=False, save_obj=None):
    pxy_t = []
    pxy_t_ = []
    if plotd == 'train':
        for t in dataset.train:
            pxy_t.append(t[-1])
            pxy_t_.append(model(t)[0])
    elif plotd == 'val':
        for t in dataset.val:
            pxy_t.append(t[-1])
            pxy_t_.append(model(t)[0])
    elif plotd == 'test':
        for t in dataset.test:
            pxy_t.append(t[-1])
            pxy_t_.append(model(t)[0])

    pxy_t_ = tf.concat(pxy_t_, axis=0)
    pxy_t = tf.concat(pxy_t, axis=0)

    max_xy, min_xy = dataset.get_min_max_pos()
    pxy_t = pxy_t * (max_xy - min_xy) + min_xy
    pxy_t_ = pxy_t_ * (max_xy - min_xy) + min_xy
    k = tf.reduce_sum(tf.subtract(pxy_t, pxy_t_) ** 2, axis=1)
    k = tf.argsort(k)
    k = k[:kfirst].numpy().tolist()
    pxy_t = pxy_t.numpy()[k]
    pxy_t_ = pxy_t_.numpy()[k]
    plt.figure(figsize=(16, 8))
    if plot_arrows:
        for y, y_ in zip(pxy_t, pxy_t_):
            plt.arrow(y_[0], y_[1], y[0] - y_[0], y[1] - y_[1])
    plt.scatter(pxy_t[:, 0], pxy_t[:, 1], marker='^', label='actual')
    plt.scatter(pxy_t_[:, 0], pxy_t_[:, 1], marker='*', label='predict')

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.scatter(pxy_t[:, 0], pxy_t_[:, 0])
    plt.subplot(1, 2, 2)
    plt.scatter(pxy_t[:, 1], pxy_t_[:, 1])