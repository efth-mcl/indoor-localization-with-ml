import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def sift_point_to_best(best_point, point, sift_dist):
    dist = np.sqrt(np.sum((point - best_point) ** 2))
    a = sift_dist / dist
    new_point = np.array([
        point[0] * a + (1 - a) * best_point[0],
        point[1] * a + (1 - a) * best_point[1]
    ])
    return new_point[0], new_point[1]


def pca_denoising(p_expls, pca_emb, pca_expls, knn, knn_pca, Dx=0, Dy=1):
    diferent_cls_ts = np.where(knn_pca.kneighbors(pca_expls[:, [Dx, Dy]])[1] != knn.kneighbors(p_expls)[1])[0]
    knn_n = knn.kneighbors(p_expls)[1]
    for ii in diferent_cls_ts:
        sift_dist = knn_pca.kneighbors(pca_expls[:, [Dx, Dy]])[0][ii][0]

        x, y = sift_point_to_best(pca_emb[knn_n[ii][0], [Dx, Dy]], pca_expls[ii, [Dx, Dy]], sift_dist)
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
    knn_pca.fit(pca_emb[:3, [Dx, Dy]], Y)

    pca_vl = pca_denoising(p_val[embidx], pca_emb, pca_vl, model.knn, knn_pca)
    pca_ts = pca_denoising(p_test[embidx], pca_emb, pca_ts, model.knn, knn_pca)

    return pca_vl, pca_ts, pca_emb, knn_pca


def n_identity_matrix(N):
    return tf.cast([[[1 if i == j and j == w else 0 for i in range(N)] for j in range(N)] for w in range(N)],
                   tf.float32)

if __name__ == "__main__":
    import doctest
    doctest.testmod()