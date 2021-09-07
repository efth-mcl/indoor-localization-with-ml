import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_psxy(model, dataset, plotd='train', maxdist=np.inf, save_obj=None):
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


def plot_psxy2(model, dataset, plotd='train', kfirst=10, plot_arrows=False, save_obj=None):
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