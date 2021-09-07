from tensorflow.keras.metrics import Metric
import tensorflow as tf
from functools import partial

class WeightedCrossEntropyWithLogits(Metric):
  """
  Object based on ` tf.nn.weighted_cross_entropy_with_logits ` with normalization parameter.

  Attributes:
    __loss: An object, weighted cross entropy with logits loss object where the weight given as partial,
    __norm: A float, normalization parameter. This attribute multiply with the outcome of __loss,
    __losssum: An object, the sumation of __loss over batch where in most common usages at the end of epoch reset to $0$.
  """
  def __init__(self, w_p, norm):
    """
    Args:
      w_p: A float, weight of loss (weighted cross entropy),
      norm: A float, normalization parameter.
    """
    super(WeightedCrossEntropyWithLogits, self).__init__(name='weighted_cross_entropy_with_logits')
    self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits,pos_weight=w_p)
    self.__norm = norm
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')

  def update_state(self, y_true, y_pred):
    """
    Update __losssum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.
    """
    self.__losssum.assign_add(tf.reduce_mean(self.__loss(y_true, y_pred)))

  def result(self):
    """
    The result of normalized loss sum. The most common usage is after the end of epoch.

    Returns:
      norm_loss: A tf_float, the normalized loss sum.
    """
    norm_loss = self.__norm * self.__losssum
    return norm_loss

  def reset_states(self):
    """
    Reset __losssum to $0$
    """
    self.__losssum.assign(0)


class MeanSquaredErrorWithLambda(Metric):
  """
  Based on ` tf.keras.metrics.MeanSquaredError ` object using a $lambda$ parameter where reduce or increase the "strength"
  of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
  and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

  Attributes:
    __loss: An object, Mean Square Error (MSE) tf Metric object,
    __losssum: An object, the sumation of __loss over batch where in most common usages at the end of epoch reset to $0$.
    __lambda: A float, a parameter where multiplied with __losssum on result step.
  """
  def __init__(self, lamda=1.0):
    """
    Args:
      lamda: float (optional), lambda parameter where using at multiplication with MSE loss.
      Default is $1.0$.
    """
    super(MeanSquaredErrorWithLambda, self).__init__(name='l2mse')
    self.__loss = tf.keras.metrics.MeanSquaredError()
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')
    self.__lambda = lamda

  def update_state(self, y_true, y_pred):
    """
    Update __losssum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.

    """
    l2loss = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(l2loss))

  def result(self):
    """
    The result of: loss multiplied with lambda. The most common usage is after the end of epoch.

    Returns:
      lambda_loss: A tf_float, loss multiplied with lambda.
    """
    lambda_loss = self.__lambda*self.__losssum
    return lambda_loss

  def reset_states(self):
    """
    Reset __losssum to $0$

    """
    self.__losssum.assign(0)

  def set_lambda(self, lamda:float):
    """
    Lamda setter.
    Args:
      lamda: A float, new lambda

    """
    self.__lambda = lamda


class CategoricalCrossEntropyWithLambda(Metric):
  """
  Based on ` tf.keras.metrics.CategoricalCrossentropy ` object using a $lambda$ parameter where reduce or increase the "strength"
  of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
  and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

  Attributes:
    __loss: An object, Categorical Cross Entropy (CCE) tf Metric object,
    __losssum: An object, the sumation of __loss over batch where in most common usages at the end of epoch reset to $0$,
    __lambda: A float, a parameter where multiplied with __losssum on result step.
  """
  def __init__(self, lamda=1.0):
    """
    Args:
      lamda: float (optional), lambda parameter where using at multiplication with CCE loss. Default is 1.0 .
    """
    super(CategoricalCrossEntropyWithLambda, self).__init__(name='cce_l')
    self.__loss = tf.keras.metrics.CategoricalCrossentropy()
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')
    self.__lambda = lamda

  def update_state(self, y_true, y_pred):
    """
    Update __losssum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.
    """
    l2loss = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(l2loss))

  def result(self):
    """
    The result of: loss multiplied with lambda. The most common usage is after the end of epoch.

    Returns:
      lambda_loss: A tf_float, loss multiplied with lambda.
    """
    lambda_loss = self.__lambda*self.__losssum
    return lambda_loss

  def reset_states(self):
    """
    Reset __losssum to $0$

    """
    self.__losssum.assign(0)

  def set_lambda(self, lamda:float):
    """
    Lambda setter.
    Args:
      lamda: A float, new lambda

    """
    self.__lambda = lamda


class LogCoshMetric(Metric):
  """
  LogCosh based on ` tf.keras.metrics.logcosh `.

  Attributes:
    __loss: An object, Logarithm Cosh tf Metric object,
    __losssum: An object, the summation of __loss over batch where in most common usages at the end of epoch reset to $0$,
  """
  def __init__(self):
    super(LogCoshMetric, self).__init__(name="logcoshmtr")
    self.__loss = tf.keras.metrics.logcosh
    self.__losssum = self.add_weight(name="losssum", initializer='zeros')

  def update_state(self, y_true, y_pred):
    """
    Update __losssum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.
    """
    logcosmtr = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(logcosmtr))

  def result(self):
    """
    Result of current loss summation over batches.

    Returns:
      losssum: A tf_float, current loss summation result.
    """
    losssum = self.__losssum
    return losssum

  def reset_states(self):
    """
    Reset __losssum to $0$.
    """
    self.__losssum.assign(0)


class MeanEuclidianError(Metric):
  """
  Mean Euclidean Error over batches. Special case for Post thesis work using min_xy where is minimum value of XY cordinates
  and maximum value of XY cordinates of given Positioning data (see Reference).

  Attributes:
     __loss: A lambda object, given y (true examples) and y_ (prediction of neural network) return the Euclidean Distance,
     __losssum: An object, the summation of __loss over batch where in most common usages at the end of epoch reset to $0$,
     __min_xy: A float, minimum value of XY coordinates,
     __max_xy: A float, maximum value of XY coordinates.

  References:
    blablabla

  """
  def __init__(self, min_xy:float, max_xy:float):
    """
    Args:
      min_xy: A float, minimum value of XY coordinates,
      max_xy: A float, maximum value of XY coordinates.
    """
    super(MeanEuclidianError, self).__init__(name='mee')
    self.__loss = lambda y, y_: tf.sqrt(tf.reduce_sum((y - y_) ** 2, axis=1))
    self.__losssum = self.add_weight(name="losssum", initializer='zeros')
    self.__min_xy, self.__max_xy = (min_xy, max_xy)

  @tf.autograph.experimental.do_not_convert
  def update_state(self, y_true, y_pred):
    """
    Update __losssum by averaging the output of __loss.

    Args:
      y_true: A tf_tensor, the true examples,
      y_pred: A tf_tensor, the predicted output of neural network.

    """
    y_true = y_true * (self.__max_xy - self.__min_xy) + self.__min_xy
    y_pred = y_pred * (self.__max_xy - self.__min_xy) + self.__min_xy
    loss = self.__loss(y_true, y_pred)
    self.__losssum.assign_add(tf.reduce_mean(loss))

  def result(self):
    """
    Result of current loss summation over batches.

    Returns:
      losssum: A tf_float, current loss summation result.
    """
    losssum = self.__losssum
    return losssum

  def reset_states(self):
    """
    Reset __losssum to $0$.
    """
    self.__losssum.assign(0)


