from typing import Union

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import copy
from functools import partial


class MetricLossBase(object):
  """
  Base of (Metric and Loss)-Base classes. This class uses 3 types for examples (input or true output) where going through
  Metric object (this type of object have a usage of performance of Neural Network). In this case the status is 2. For
  Loss object (this type of object have a usage of calculate the gradient of top of Neural Network). In this case the status
  is 1. Finally, for input examples, the status is 0. The reason of tow type of status (1 & 2) is that in some cases such as
  Zero Shot learning the top output of Neural Network could be a vector where this vector goes through post process into classifier.
  In this case only use loss status for top output of Neural Network and input status.
  0 -> input
  1 -> output loss
  2 -> output metric
  """
  def __init__(self):
    pass
  
  def status_idxs(self, type:int, status:list):
    """
    Args:
      type: A int, type of status (0, 1 or 2),
      status: A list, nested list of status per data in example.

    Returns:
      idxs: A list, list with indexes from `status` where mach with `type`

    Examples:
      Give an example usage where the example have 3 data types (input, output (1st) with loss status type, output (2nd)
      with metric & loss status type).

      >>> status = [[0], [1], [1,2]]
      >>> # 1st case
      >>> type = 0
      >>> print(self.status_idxs(type, status))
      [0]
      >>> # 2nd case
      >>> type = 1
      >>> print(self.status_idxs(type, status))
      [1, 2]
      >>> # 3rd case
      >>> type = 2
      >>> print(self.status_idxs(type, status))
      [2]
    """
    idxs = []
    for i, st in enumerate(status):
      if type in st:
        idxs.append(i)
    return idxs
  
  def get_batch_by_indxs(self, batch:Union[list, tuple], idxs:list):
    """

    Args:
      batch: A list or tuple, subset of data examples,
      idxs: A list, list of indexes.

    Returns:
      batch_by_indxs: A tuple, specific data examples based on idxs.
    """

    batch_by_indxs = tuple([batch[i] for i in idxs])
    return batch_by_indxs


class MetricBase(MetricLossBase):
  """
  Multiple metric avenging manipulation. Create a list of metrics such as `tf.keras.metric.Metric`.

  Attributes:
    __mtr: A list, list of metric objects based on ` tf.keras.metric.Metrics `,
    __mtr_idxs: A list, indexes list of status with for output type status,
    __mtr_select: A list, list of indexes of __mtr,
    __input_idxs: A list, indexes list of input data examples,
    __callfn: An object, model call method.
  """
  def __init__(self, model, mtrs_obj, status, mtr_select, status_out_type=2):
    """

    Args:
      model: An object, based on ` tf.Module ` or ` tf.keras.model.Model `,
      mtrs_obj: A list, list of metric objects based on ` tf.keras.metric.Metrics `,
      status: A list, nested list of status per data of examples,
      mtr_select: A list, list of indexes of mtrs_obj,
      status_out_type: int (optional), type of output data of examples. Default is 2.
    """
    super(MetricBase, self).__init__()
    self.__mtr = mtrs_obj
    self.__mtr_idxs = self.status_idxs(status_out_type, status)
    self.__mtr_select = mtr_select
    self.__input_idxs = self.status_idxs(0,status)

    if hasattr(model,'call'):
      self.__callfn = model.call
    else:
      self.__callfn = model.__call__

  def metric_dataset(self, dataset:tf.data.Dataset):
    """
    Calculate the average metric over all metrics depends on dataset (examples) status structure.

    Args:
      dataset: A tf.data.Dataset,
    Returns:
      mtr: tf float, average metric of all metrics summation.

    """
    for batch in dataset:
      inputs = self.get_batch_by_indxs(batch, self.__input_idxs)
      true_outputs = self.get_batch_by_indxs(batch,self.__mtr_idxs)
      predict_outputs = self.__callfn(inputs)
      self.metric_batch(true_outputs, predict_outputs)

    mtr = 0
    for metric in self.__mtr:
      mtr += metric.result().numpy()
      metric.reset_states()

    mtr = mtr/tf.cast(len(self.__mtr),tf.float32)
    return mtr

  def metric_batch(self, true_outputs:Union[list, tuple], predict_outputs:Union[list, tuple]):
    """
    Update all metrics over batch.

    Args:
      true_outputs: A list, true outputs list of tensors,
      predict_outputs: A list, predicted outputs list of tensors.

    """
    for i, ms in enumerate(self.__mtr_select):
      self.__mtr[ms].update_state(true_outputs[i], predict_outputs[i])


class LossBase(MetricLossBase):
  """
  Multiple loss avenging manipulation. Create a list of metrics such as `tf.keras.losses.Loss`.

  Attributes:
    __loss: A list, list of losses objects based on ` tf.keras.loeses.Loss `,
    __loss_idxs: A list, indexes list of status with for output type status,
    __loss_select: A list, list of indexes of __loss,
    __input_idxs: A list, indexes list of input data examples,
    __callfn: An object, model call method.
  """
  def __init__(self, model, losses_obj, status, loss_select):
    """

    Args:
      model: An object, based on ` tf.Module ` or ` tf.keras.model.Model `,
      losses_obj: A list, list of loss objects based on ` tf.keras.losses.Loss `,
      status: A list, nested list of status per data of examples,
      loss_select: A list, list of indexes of losses_obj.
    """
    super(LossBase, self).__init__()
    self.__loss = losses_obj
    self.__loss_idxs = self.status_idxs(1,status)
    self.__loss_select = loss_select
    self.__input_idxs = self.status_idxs(0,status)

    if hasattr(model,'call'):
      self.__callfn = model.call
    else:
      self.__callfn = model.__call__

  
  def loss_batch(self, batch:Union[list, tuple]):
    """
    Compute loss over all losses objects (__loss list) using summation.
    Args:
      batch: A list or tuple, subset of data examples.

    Returns:
      loss: tf float, the summation of all losses.
    """
    loss = 0
    inputs = self.get_batch_by_indxs(batch,self.__input_idxs)
    predict_outputs = self.__callfn(inputs)
    true_outputs = self.get_batch_by_indxs(batch,self.__loss_idxs)
    for i, ls in enumerate(self.__loss_select):
      loss += self.__loss[ls](true_outputs[i], predict_outputs[i])
        
    
    return loss


class EarlyStoping(object):
  """
  Attributes:
    es_strategy: A str, early stopping strategy. {'first_drop' or 'patience'},
    es_metric: A srt, metric where strategy follow. {'{train, val or test}_{cost or score}'},
    __metric_max: A float, this created if strategy is 'first_drop',
    __save_weights_obj: An object (optional), tf.models.Model or tf.Module object save_weights with partial input (path). Default is None.

    Examples:
        # patience strategy example:
        >>> entry = {'es_strategy':'patience', 'es_metric':'val_cost', 'es_min_delta': 1e-3, 'es_patience': 10 }
        >>> ES = EarlyStoping(entry)
        >>> print(ES.es_strategy)
        'patience'

        # first drop strategy
        >>> entry = {'es_strategy':'first_drop', 'es_metric':'val_score'}
        >>> ES = EarlyStoping(entry)
        >>> print(ES.es_strategy)
        'first_drop'

        # save weights object example
        >>> # entry = {'es_strategy':'first_drop', 'es_metric':'val_score'}
        >>> # save_weights_obj = partial(self.save_weights, "path/to/weights.tf")
        >>> # ES = EarlyStoping(entry, save_weights_obj)
  """
  def __init__(self, entries, save_weights_obj=None):
    """

    Args:
      entries: A dict, dictionary of class attributes based on early stopping (es) strategy
      save_weights_obj: An object (optional), tf.keras.models.Model or tf.Module object save_weights with partial input (path). Default is None.

    """
    self.es_strategy = None
    self.es_metric = None
    self.__dict__.update(entries)
    if self.es_strategy == 'first_drop' and self.es_metric is not None:
      self.__metric_max = 0
    self.__save_weights_obj = save_weights_obj
      
    
  def check_stop(self, h):
    """
    Application of early stopping strategy based on history (h). For `first drop` strategy training stops if the caption
    metric drops for first time. For `patience` strategy training stops when after a number of epochs the metric don't change,
    based on a `delta` range where the metric don't change.

    Args:
      h: A dict, history dictionary

    Returns:
      A bool, True or False based on early stopping strategy.
    """
    if len(h[self.es_metric])>0:
      if self.es_strategy is not None and self.es_metric is not None:
        sub_h = h[self.es_metric]
        if self.es_strategy == 'first_drop':
          if sub_h[-1] > self.__metric_max:
            self.__metric_max = sub_h[-1]
            self.__save_weights()
          elif self.__metric_max > 0 and sub_h[-1] < self.__metric_max:
            return True
          return False
        elif self.es_strategy == 'patience':
          if len(sub_h) >= self.es_patience:
            if abs(sub_h[-1] - sub_h[-self.es_patience]) < self.es_min_delta:
              return True
            else:
              return False
          return False


  def __save_weights(self):
    """
    Activate the save_weights (tf.keras.models.Model or tf.Module).
    """
    if self.__save_weights_obj is not None:
      self.__save_weights_obj()


class Trainer(object):
  """
  Training options: early stopping and gradient clipping. Keep learning history and give a verity of optimizers.

  Attributes:
    __optimizer: An optimizer object, options: {'SGD', 'Adagrad', 'Adadelta', 'Adam'},
    __clip_norm: A float, gradient clipping (normalization method),
    __epochs_cnt: An int, epochs counter,
    __early_stop: A dict, early stopping attributes on dictionary. Default is None.
    __score_mode: A bool, this boolean have use of switch the mode of zero shot nn model output (categorical (score) 'True' or vector 'False'),
    history: A dict, history per metric ({train, val, test}_{score, cost} & harmonic_score) for epoch.
  """
  def __init__(self, early_stop_vars=None, save_weights_obj=None, optimizer="SGD", learning_rate=1e-4, clip_norm=0.0):
    """

    Args:
      early_stop_vars: A dict (optional), early stopping attributes on dictionary . Default is None,
      save_weights_obj: An object (optional), tf.models.Model or tf.Module object save_weights with partial input (path). Default is None.
      optimizer: A str (optional), optimizer options: {'SGD', 'Adagrad', 'Adadelta', 'Adam'}. Default is `SGD`,
      learning_rate: A float (optional), learning rate of optimizer. Default is 1e-4,
      clip_norm: A float (optional). Default is 0.0 .
    """
    if optimizer == "SGD":
      self.__optimizer = SGD(learning_rate, momentum=0.98)
    elif optimizer == "Adagrad":
      self.__optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate,initial_accumulator_value=1e-3,epsilon=1e-7)
    elif optimizer == "Adadelta":
      self.__optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate, rho=0.98)
    elif optimizer == "Adam":
      self.__optimizer = Adam(learning_rate=learning_rate, amsgrad=True)

    self.__clip_norm = clip_norm
    

    self.__early_stop = None
    if early_stop_vars is not None:
      self.__early_stop = EarlyStoping(early_stop_vars, save_weights_obj)
    self.__epochs_cnt = 0
    self.history = {
      'train_cost': [],
      'train_score': [],
      'val_cost': [],
      'val_score': [],
      'test_cost': [],
      'test_score': [],
      'harmonic_score': []
    }

  def set_lr_rate(self, lr):
    """
    Learning rate setter.

    Args:
      lr: A float, new learning rate.

    """
    self.__optimizer.lr.assign(lr)

  def set_clip_norm(self, clip_norm):
    """
    Gradient clipping setter.

    Args:
      clip_norm: A float, new gradient clipping (normalization method).
    """
    self.__clip_norm = clip_norm

  def set_early_stop(self, early_stop_vars):
    """
    Early stopping setter.
    Args:
      early_stop_vars: A dict, new early stopping parameters.

    """
    self.__early_stop = EarlyStoping(early_stop_vars, None)

  def train(self, dataset_train, epochs=10, dataset_val=None,dataset_test=None, history_learning_process=True):

    def print_add_history(dataset, dataset_type='train'):
      self.set_score_mode(False)
      cost_mtr = self.cost_mtr.metric_dataset(dataset)

      self.history['{}_cost'.format(dataset_type)].append(
        float(cost_mtr)
      )
      print('{}_cost: {}'.format(dataset_type, cost_mtr), end='')
      self.set_score_mode(True)
      if hasattr(self, 'score_mtr'):
        score_mtr = self.score_mtr.metric_dataset(dataset)

        self.history['{}_score'.format(dataset_type)].append(
          float(score_mtr)
        )
        print(', {}_score: {}'.format(dataset_type, score_mtr))

        return cost_mtr, score_mtr
      else:
        print()
        return cost_mtr, None

    for epoch in range(epochs):
      # super hard code
      self.__epochs_cnt += 1
      self.set_score_mode(False)
      for batch in dataset_train:
        with tf.GradientTape() as tape:
          cost_loss = self.cost_loss.loss_batch(batch)
        grads = tape.gradient(cost_loss, self.trainable_variables)
        if self.__clip_norm > 0:
          grads = [(tf.clip_by_norm(grad, clip_norm=self.__clip_norm)) for grad in grads]
        self.__optimizer.apply_gradients(zip(grads, self.trainable_variables))

      print ('Epoch {} finished'.format(self.__epochs_cnt))
      if history_learning_process:
        self.set_score_mode(False)
        train_cost_mtr, train_score_mtr = print_add_history(dataset_train, dataset_type='train')
        if dataset_val is not None:
          val_cost_mtr, val_score_mtr = print_add_history(dataset_val, dataset_type='val')
        if dataset_test is not None:
          test_cost_mtr, test_score_mtr = print_add_history(dataset_test, dataset_type='test')

        if dataset_val is not None and dataset_test is not None and 'test_score_mtr' in locals() and 'val_score_mtr' in locals():
          harmonic_score = 2*test_score_mtr*val_score_mtr/(test_score_mtr+val_score_mtr)

          self.history['harmonic_score'].append(
            float(harmonic_score)
          )
          print('harmonic score: {}'.format(harmonic_score))

        if self.__early_stop is not None:
          if self.__early_stop.check_stop(copy.deepcopy(self.history)):
            print('Stop Training')
            break



class BaseNeuralNetwork(Trainer):
  """
  Base of Neural Network models. See examples on `models.py` .

  Attributes:
    __status: A list, a nested list for input/output of model (see MetricLossBase for details),
    __score_mode: A bool, this boolean have use of switch the mode of zero shot nn model output (categorical (score) 'True' or vector 'False').
  """
  def __init__(self, status, early_stop_vars=None, weights_outfile=None, optimizer="SGD", learning_rate=1e-4):
    """

    Args:
      status: A list, a nested list for input/output of model (see MetricLossBase for details),
      early_stop_vars: A dict (optional), early stopping attributes on dictionary . Default is None,
      weights_outfile: A list (optional), a list of sub-paths. The first sub-path is the root folder for weights and
                      the second sub-path is the name of the best weights. All weights file type is `.tf`,
      optimizer: A str (optional), the optimizer where use. Default is `SGD`,
      learning_rate: A float (optional), the learning rate of optimizer. Default is `1e-4`.
    """
    save_weights_obj = None
    if weights_outfile is not None:
      save_weights_obj = partial(self.save_weights, "{}/weights/weights_best_{}.tf".format(weights_outfile[0], weights_outfile[1]))
    super(BaseNeuralNetwork, self).__init__(early_stop_vars, save_weights_obj, optimizer, learning_rate)

    self.__status = status
    self.__score_mode = False


  def get_score_mode(self):
    """
    Score mode getter.
    """
    return self.__score_mode


  def set_score_mode(self, score_mode):
    """
    Score mode setter
    Args:
      score_mode: A bool, this boolean have use of switch the mode of zero shot nn model output (categorical (score) 'True' or vector 'False').

    """
    self.__score_mode = score_mode
  

  def get_results(self, dataset, score_mode=False):

    def forloop(dataset_exmpl):
        in_idxs = self.cost_mtr.status_idxs(0, self.__status)
        tr_out_idxs = self.cost_mtr.status_idxs(2, self.__status)
        for batch in dataset_exmpl:
            inputs = self.cost_mtr.get_batch_by_indxs(batch, in_idxs)
            outs = self.cost_mtr.get_batch_by_indxs(batch, tr_out_idxs)
            if hasattr(self,"__call__"):
              predict = self.__call__(inputs)
            elif hasattr(self,"call"):
              predict = self.call(inputs)
        return inputs, outs, predict
    
    self.set_score_mode(score_mode)
    inputs_train, outs_train, predict_train = forloop(dataset.train)
    inputs_val, outs_val, predict_val = forloop(dataset.val)
    inputs_test, outs_test, predict_test = forloop(dataset.test)
    self.set_score_mode(False)

    return (inputs_train, outs_train, predict_train), (inputs_val, outs_val, predict_val), (inputs_test, outs_test, predict_test)