import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlb


def pca_denoising_figure(pca_vl, pca_ts, pca_emb, knn_pca, zlabels, pca_emb_idxs= [0,1,2], save_obj=None):
    """
    Draw the knn_pca spaces, plot target embeddings, predicted validation (seen) embeddings and test (unseen) embeddings.

    Args:
        pca_vl: A ndarray, pca denoised validation embeddings where predicted,
        pca_ts: A ndarray, pca denoised test embeddings where predicted,
        pca_emb: A ndarray, pca denoised true embeddings,
        knn_pca: A KNeighborsClassifier object, trained by pca_emb,
        zlabels: A list, list of true embedding labels,
        pca_emb_idxs: A list (Optional), list of subpaths saving figure. Default is [0,1,2],
        save_obj: A list (Optional), list of subpaths saving figure. Default is None.

    """
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
    xx, yy = np.meshgrid(xlin, ylin)
    knn_space = np.argmax(knn_pca.predict(np.c_[xx.ravel(), yy.ravel()]), axis=1)
    knn_space = knn_space.reshape(xx.shape)

    Dx = 0
    Dy = 1

    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(pca_emb[pca_emb_idxs, Dx], pca_emb[pca_emb_idxs, Dy], 'o', label='embs', markersize=15)
    plt.plot(pca_vl[:, Dx], pca_vl[:, Dy], '*', label='val', markersize=10)
    plt.plot(pca_ts[:, Dx], pca_ts[:, Dy], '*', label='test', markersize=10)

    for (v, l) in zip(pca_emb, zlabels[:3]):
        plt.text(v[Dx], v[Dy], l, fontsize=20)

    ax.contourf(xx, yy, knn_space, cmap=plt.get_cmap('tab20c'), levels=2)
    plt.legend()
    if save_obj is not None:
        plt.savefig('{}/{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()



def history_figure(history, figsize=(16, 8), legend_fontsize=18, axes_label_fondsize=22, ticks_fontsize=16, save_obj=None):
    f"""

    Args:
        history: A dict, train, validation test and harmonic mean history of model. The model interference ` thesispack.base.BaseNueralNetwork ` 
                 object,
        figsize: A tuple or list, figure size of x and y axis,
        legend_fontsize: A int,
        axes_label_fondsize: A int,
        ticks_fontsize: A int,
        save_obj: A list, list of path and name of saving figure.

    """
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
        plt.savefig('{}/{}.eps'.format(*save_obj), format='eps')
    else:
        plt.show()


def print_confmtx(model, dataset, lerninfo: str, indexes_list: list):
    """
    Print confusion matrix per examples (train, test, validation)
    Args:
        model: An object, The model interference ` thesispack.base.BaseNueralNetwork ` object,
        dataset: An object,
        lerninfo: A str,
        indexes_list: A list.
    """
    def confmtx(y, yhat):
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