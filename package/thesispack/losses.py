import tensorflow as tf
from functools import partial



class WeightedCrossEntropyWithLogits(object):
    def __init__(self, w_p, norm):
        self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=w_p)
        self.__norm = norm

    def __call__(self, y_true, y_pred):
        loss = self.__norm * tf.reduce_mean(self.__loss(y_true, y_pred))
        return loss


class MeanSquaredErrorWithLambda(object):
    def __init__(self, lamda):
        self.__lambda = lamda
        self.__loss = tf.keras.losses.MeanSquaredError()

    def __call__(self, y_true, y_pred):
        return self.__lambda * self.__loss(y_true, y_pred)

    def set_lambda(self, lamda):
        self.__lambda = lamda


class CategoricalCrossentropyWithLambda(object):
    def __init__(self, lamda):
        self.__lambda = lamda
        self.__loss = tf.keras.losses.CategoricalCrossentropy()

    def __call__(self, y_true, y_pred):
        return self.__lambda * self.__loss(y_true, y_pred)

    def set_lambda(self, lamda):
        self.__lambda = lamda
