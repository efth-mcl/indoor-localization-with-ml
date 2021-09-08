import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def sift_point_to_best(target_point, point, sift_dist):
    """
    Move a point to target point given a distance. Based on Jensen's inequality formula.

    Args:
        target_point: A ndarray or tensor, the target point of pca,
        point: A ndarray or tensor, point of pca,
        sift_dist: A float, distance where point will sift to new one.

    Returns:
        new_points: A tuple, a couple of new updated points.

    References:
        https://en.wikipedia.org/wiki/Jensen%27s_inequality
    """

    dist = np.sqrt(np.sum((point - target_point) ** 2))
    a = sift_dist / dist
    new_point = np.array([
        point[0] * a + (1 - a) * target_point[0],
        point[1] * a + (1 - a) * target_point[1]
    ])
    new_points = (new_point[0], new_point[1])
    return new_points


def pca_denoising(p_expls, pca_emb_expls, pca_emb_pred, knn, knn_pca, Dx=0, Dy=1):
    """
    PCA denosing method.

    Args:
        p_expls: A ndarray or tensor, high dimensional data embeddings of target points,
        pca_emb_expls: A ndarray or tensor, low dimensional pca data embeddings of target points,
        pca_emb_pred: A ndarray or tensor, low dimensional pca data embeddings of prediction.
        knn: A KNeighborsClassifier object, knn model trained by p_expls,
        knn_pca: A KNeighborsClassifier object, knn model trained by pca_emb_expls,
        Dx: A int, index dimension of x axis,
        Dy: A int, index dimension of y axis.

    Returns:
        pca_emb_pred: A ndarray or tensor, low dimensional updated pca data embeddings of predition.

    See Also:
        After pca process the real distance between pca examples and pca embeddings have noise by dimension reduction.
    """
    diferent_cls_ts = np.where(knn_pca.kneighbors(pca_emb_pred[:, [Dx, Dy]])[1] != knn.kneighbors(p_expls)[1])[0]
    knn_n = knn.kneighbors(p_expls)[1]
    for ii in diferent_cls_ts:
        sift_dist = knn_pca.kneighbors(pca_emb_pred[:, [Dx, Dy]])[0][ii][0]

        x, y = sift_point_to_best(pca_emb_expls[knn_n[ii][0], [Dx, Dy]], pca_emb_pred[ii, [Dx, Dy]], sift_dist)
        pca_emb_pred[ii, Dx] = x
        pca_emb_pred[ii, Dy] = y

    return pca_emb_pred


def pca_denoising_preprocessing(model, dataset, Z, Y, embidx=0, pca_emb_idxs=[0, 1, 2]):
    f"""

    Args:
        model: An object, this object interference ` algomorphism.BaseNeuralNetowk ` objcet & tensorflow.{tf.Module, tf.keras.models.Model},
        dataset: An object, this object has train, val and test {tf.data.Dataset.from_tensor_slices},
        Z: A tensor, true embeddings,
        Y: A tensor, predicted embeddings,
        embidx: (Optional) . Default is 0,
        pca_emb_idxs: (Optional) . Default is [0,1,2].

    Returns:
        pca_vl: A ndarray, pca denoised validation embeddings where predicted,
        pca_ts: A ndarray, pca denoised test embeddings where predicted,
        pca_emb: A ndarray, pca denoised true embeddings,
        knn_pca: A KNeighborsClassifier object.
    """
    (_, _, p_train), (_, _, p_val), (_, _, p_test) = model.get_results(dataset, False)

    pca = PCA(n_components=2)
    pca.fit(Z)

    pca_emb = pca.transform(Z)
    pca_vl = pca.transform(p_val[embidx])
    pca_ts = pca.transform(p_test[embidx])

    Dx = 0
    Dy = 1
    knn_pca = KNeighborsClassifier(1)
    knn_pca.fit(pca_emb[pca_emb_idxs, [Dx, Dy]], Y)

    pca_vl = pca_denoising(p_val[embidx], pca_emb, pca_vl, model.knn, knn_pca)
    pca_ts = pca_denoising(p_test[embidx], pca_emb, pca_ts, model.knn, knn_pca)

    return pca_vl, pca_ts, pca_emb, knn_pca

def n_identity_matrix(N):
    """
    Return a 3D identity matrix.
    Args:
        N: A int, size of 3d matrix.

    Returns:
        A tf_tensor, 3D identity matrix.
    """
    return tf.cast([[[1 if i == j and j == w else 0 for i in range(N)] for j in range(N)] for w in range(N)],
                   tf.float32)