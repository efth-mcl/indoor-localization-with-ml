import tensorflow as tf
from functools import partial



class WeightedCrossEntropyWithLogits(object):
    """
     Object based on ` tf.nn.weighted_cross_entropy_with_logits ` with normalization parameter.

     Attributes:
        __loss: An object, weighted cross entropy with logits loss object where the weight given as partial,
       __norm: A float, normalization parameter. This attribute multiply with the outcome of __loss,
    """
    def __init__(self, w_p, norm):
        """
        Args:
            w_p: A float, weight of loss (weighted cross entropy),
            norm: A float, normalization parameter.
        """
        self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=w_p)
        self.__norm = norm

    def __call__(self, y_true, y_pred):
        """
        Call function.
        Args:
            y_true: A tf_tensor, the true examples,
            y_pred: A tf_tensor, the predicted output of neural network.

        Returns:
            loss: A tf_float, mean __loss multiplied by __norm
        """
        loss = self.__norm * tf.reduce_mean(self.__loss(y_true, y_pred))
        return loss


class MeanSquaredErrorWithLambda(object):
    """
      Based on ` tf.keras.losses.MeanSquaredError ` object using a $lambda$ parameter where reduce or increase the "strength"
      of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
      and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

      Attributes:
        __loss: An object, Mean Square Error (MSE) tf Loss object,
        __lambda: A float, a parameter where multiplied with __loss on result step.
    """
    def __init__(self, lamda:float=1.0):
        """
        Args:
          lamda: float (optional), lambda parameter where using at multiplication with MSE loss.
          Default is $1.0$.
        """
        self.__lambda = lamda
        self.__loss = tf.keras.losses.MeanSquaredError()

    def __call__(self, y_true, y_pred):
        """
       Call function.

       Args:
           y_true: A tf_tensor, the true examples,
           y_pred: A tf_tensor, the predicted output of neural network.

       Returns:
           loss: A tf_float, mean __loss multiplied by __lambda
        """
        return self.__lambda * self.__loss(y_true, y_pred)

    def set_lambda(self, lamda):
        self.__lambda = lamda


class CategoricalCrossEntropyWithLambda(object):
    """
    Based on ` tf.keras.losses.CategoricalCrossentropy ` object using a $lambda$ parameter where reduce or increase the "strength"
    of gradient at backpropagation step. A common usage is where adding two types of loss and this loss have different gradient
    and want to balance the gradients. $Lambda$ could be choice with trials or more wise with hyper-parameter search methods.

    Attributes:
        __loss: An object, Categorical Cross Entropy (CCE) tf Loss object,
        __lambda: A float, a parameter where multiplied with __loss on result step.
    """
    def __init__(self, lamda:float=1.0):
        """
        Args:
            lamda: float (optional), lambda parameter where using at multiplication with CCE loss.
            Default is $1.0$.
        """
        self.__lambda = lamda
        self.__loss = tf.keras.losses.CategoricalCrossentropy()

    def __call__(self, y_true, y_pred):
        """
        Call function.

        Args:
            y_true: A tf_tensor, the true examples,
            y_pred: A tf_tensor, the predicted output of neural network.

        Returns:
            loss: A tf_float, mean __loss multiplied by __lambda
        """
        loss = self.__lambda * self.__loss(y_true, y_pred)
        return loss

    def set_lambda(self, lamda):
        """
        Lambda setter.
        Args:
            lamda: a float, new lambda parameter.
        """
        self.__lambda = lamda
