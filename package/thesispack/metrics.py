from tensorflow.keras.metrics import Metric
import tensorflow as tf
from functools import partial

class WeightedCrossEntropyWithLogits(Metric):

  def __init__(self, w_p, norm):
    super(WeightedCrossEntropyWithLogits, self).__init__(name='weighted_cross_entropy_with_logits')
    self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits,pos_weight=w_p)
    self.__norm = norm
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.__losssum.assign_add(tf.reduce_mean(self.__loss(y_true, y_pred)))

  def result(self):
    return self.__norm * self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)


class MeanSquaredErrorWithLambda(Metric):
  def __init__(self, lamda=1):
    super(MeanSquaredErrorWithLambda, self).__init__(name='l2mse')
    self.__loss = tf.keras.metrics.MeanSquaredError()
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')
    self.__lambda = lamda

  def update_state(self, y_true, y_pred):
    l2loss = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(l2loss))

  def result(self):
    return self.__lambda*self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)

  def set_lambda(self, lamda):
      self.__lambda = lamda


class CategoricalCrossentropyWithLambda(Metric):
  def __init__(self, lamda=1):
    super(CategoricalCrossentropyWithLambda, self).__init__(name='l2cce')
    self.__loss = tf.keras.metrics.CategoricalCrossentropy()
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')
    self.__lambda = lamda

  def update_state(self, y_true, y_pred):
    l2loss = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(l2loss))

  def result(self):
    return self.__lambda*self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)

  def set_lambda(self, lamda):
      self.__lambda = lamda


class LogCosMetric(Metric):
  def __init__(self):
    super(LogCosMetric, self).__init__(name="logcosmtr")
    self.__loss = tf.keras.metrics.logcosh
    self.__losssum = self.add_weight(name="losssum", initializer='zeros')

  def update_state(self, y_true, y_pred):
    logcosmtr = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(logcosmtr))

  def result(self):
    return self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)


class MeanEuclidianError(Metric):
  def __init__(self, min_xy, max_xy):
    super(MeanEuclidianError, self).__init__(name='mee')
    self.__loss = lambda y, y_: tf.sqrt(tf.reduce_sum((y - y_) ** 2, axis=1))
    self.__losssum = self.add_weight(name="losssum", initializer='zeros')
    self.__min_xy, self.__max_xy = (min_xy, max_xy)

  @tf.autograph.experimental.do_not_convert
  def update_state(self, y_true, y_pred):
    y_true = y_true * (self.__max_xy - self.__min_xy) + self.__min_xy
    y_pred = y_pred * (self.__max_xy - self.__min_xy) + self.__min_xy
    loss = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(loss))

  def result(self):
    return self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)