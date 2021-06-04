
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.activations import tanh
import tensorflow as tf
from tensorflow.keras import initializers
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import copy
from functools import partial

from thesispack.models import GCN, IP

from .methods import n_identity_matrix


global LAMBDA, EQ1
EQ1 = False
LAMBDA = 1
# 0 input
# 1 output loss
# 2 output mtr
class MetricLossBase(object):
  def __init__(self, status):
    # auto tha eprepe na klironomite apo tis main classes
    # self.__input_idxs = self.status_idxs(0,status)
    pass
  
  def status_idxs(self, tepy, status):
    idxs = []
    for i, st in enumerate(status):
      if tepy in st:
        idxs.append(i)
    return idxs
  
  def get_batch_by_indxs(self, batch, idxs):
    return tuple([batch[i] for i in idxs])


class MetricBase(MetricLossBase):
  def __init__(self, model, mtrs_obj, status, mtr_select, status_out_key=2):
    super(MetricBase, self).__init__(status)
    self.__mtr = mtrs_obj
    self.__mtr_idxs = self.status_idxs(status_out_key, status)
    self.__mtr_select = mtr_select
    self.__input_idxs = self.status_idxs(0,status)

    if hasattr(model,'call'):
      self.__callfn = model.call
    else:
      self.__callfn = model.__call__

  def metric_dataset(self, dataset):
    for batch in dataset:
      inputs = self.get_batch_by_indxs(batch, self.__input_idxs)
      true_outputs = self.get_batch_by_indxs(batch,self.__mtr_idxs)
      predict_outputs = self.__callfn(inputs)
      self.metric_batch(true_outputs, predict_outputs)

    mtr = 0
    for metric in self.__mtr:
      mtr += metric.result().numpy()
      metric.reset_states()
    return mtr/tf.cast(len(self.__mtr),tf.float32)

  def metric_batch(self, true_outputs, predict_outputs):
    for i, ms in enumerate(self.__mtr_select):
      self.__mtr[ms].update_state(true_outputs[i], predict_outputs[i])


class LossBase(MetricLossBase):

  def __init__(self, model, losses_obj, status, loss_select):
    super(LossBase, self).__init__(status)
    self.__loss = losses_obj
    self.__loss_idxs = self.status_idxs(1,status)
    self.__loss_select = loss_select
    self.__input_idxs = self.status_idxs(0,status)

    if hasattr(model,'call'):
      self.__callfn = model.call
    else:
      self.__callfn = model.__call__

  
  def loss_batch(self, batch):
    loss = 0
    inputs = self.get_batch_by_indxs(batch,self.__input_idxs)
    predict_outputs = self.__callfn(inputs)
    true_outputs = self.get_batch_by_indxs(batch,self.__loss_idxs)
    for i, ls in enumerate(self.__loss_select):
      loss += self.__loss[ls](true_outputs[i], predict_outputs[i])
        
    
    return loss


class EarlyStoping(object):
  def __init__(self, entries, save_weights_obj=None):
    self.es_strategy = None
    self.es_metric = None
    self.__dict__.update(entries)
    if self.es_strategy == 'first_drop' and self.es_metric is not None:
      self.__metric_max = 0
    self.__save_weights_obj = save_weights_obj
      
    
  def check_stop(self, h):
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
    if self.__save_weights_obj is not None:
      self.__save_weights_obj()


class Trainer(object):
  def __init__(self, early_stop_vars=None, save_weights_obj=None, optimizer="SGD", learning_rate=1e-4, clip_norm=2.0):
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
    self.__optimizer.lr.assign(lr)

  def set_clip_norm(self, clip_norm):
    self.__clip_norm = clip_norm

  def set_early_stop(self, early_stop_vars):
    self.__early_stop = EarlyStoping(early_stop_vars, None)

  def train(self, dataset, epochs=10, dataset_val=None,dataset_test=None, history_learning_process=True):
    # we use an print_return_history flag how to not use this maybe use class or curried function
    # use something to not use 'if' again (like class)!
    for epoch in range(epochs):
      # super hard code
      self.__epochs_cnt += 1
      self.set_score_mode(False)
      for batch in dataset:
        with tf.GradientTape() as tape:
          cost_loss = self.cost_loss.loss_batch(batch)
        grads = tape.gradient(cost_loss, self.trainable_variables)
        if self.__clip_norm > 0:
          grads = [(tf.clip_by_norm(grad, clip_norm=self.__clip_norm)) for grad in grads]
        self.__optimizer.apply_gradients(zip(grads, self.trainable_variables))

      print ('Epoch {} finished'.format(self.__epochs_cnt))
      if history_learning_process:
        self.set_score_mode(False)
        train_cost_mtr = self.cost_mtr.metric_dataset(dataset)
        self.set_score_mode(True)
        train_score_mtr = self.score_mtr.metric_dataset(dataset)
        
        self.history['train_cost'].append(
            float(train_cost_mtr)
        )
        self.history['train_score'].append(
            float(train_score_mtr)
        )
        print('train_cost: {}, train_score: {}'.format(train_cost_mtr, train_score_mtr))

        if dataset_val is not None:
          # this is super hard code
          self.set_score_mode(False)
          val_cost_mtr = self.cost_mtr.metric_dataset(dataset_val)
          self.set_score_mode(True)
          val_score_mtr = self.score_mtr.metric_dataset(dataset_val)

          self.history['val_cost'].append(
             float(val_cost_mtr)
          )
          self.history['val_score'].append(
              float(val_score_mtr)
          )
          print('val_cost: {}, val_score: {}'.format(val_cost_mtr, val_score_mtr))
        if dataset_test is not None:
          self.set_score_mode(False)
          test_cost_mtr = self.cost_mtr.metric_dataset(dataset_test)
          self.set_score_mode(True)
          test_score_mtr = self.score_mtr.metric_dataset(dataset_test)

          self.history['test_cost'].append(
            float(test_cost_mtr)
          )
          self.history['test_score'].append(
            float(test_score_mtr)
          )
          print('test_cost: {}, test_score: {}'.format(test_cost_mtr, test_score_mtr))

        if dataset_val is not None and dataset_test is not None:
          harmonic_score = 2*test_score_mtr*val_score_mtr/(test_score_mtr+val_score_mtr)

          self.history['harmonic_score'].append(
            float(harmonic_score)
          )
          print('harmonic score: {}'.format(harmonic_score))

        if self.__early_stop is not None:
          if self.__early_stop.check_stop(copy.deepcopy(self.history)):
            print('Stop Training')
            break

      # try:
      #   # hard code early stoping
      #   if float(test_score_mtr) == 1.0:
      #     global LAMDA, EQ1
      #     if not EQ1:
      #       EQ1 = True
      #       LAMDA = 1e-1
      #       print("update lamda to:", LAMDA)
      #
      #   if float(harmonic_score) > 0.95:
      #     break
      # except Exception as e:
      #   pass


# 0 input
# 1 output | loss
# 2 output | mtr
class BaseNeuralNetwork(Trainer):
  def __init__(self, status, early_stop_vars=None, weights_outfile=None, optimizer="SGD", learning_rate=1e-4):
    save_weights_obj = None
    if weights_outfile is not None:
      save_weights_obj = partial(self.save_weights, "{}/weights/weights_best_{}.tf".format(weights_outfile[0], weights_outfile[1]))
    super(BaseNeuralNetwork, self).__init__(early_stop_vars, save_weights_obj, optimizer, learning_rate)

    self.__status = status
    self.__score_mode = False
  

  def get_score_mode(self):
    return self.__score_mode


  def set_score_mode(self, score_mode):
    self.__score_mode = score_mode
  

  def get_results(self, dataset, score_mode=False):

    # this is simple for 1 loop
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


#-------------------------------#
#- Pre codes -------------------#
#-------------------------------#
class GAE(tf.Module):
  def __init__(self,ft_number):
    super(GAE,self).__init__(name='gae')

    # self.gcn1 = GCN(ft_number,128,'relu')
    # self.gcn2 = GCN(128,256)

    self.gcn1 = GCN(ft_number,64,'relu')
    self.gcn2 = GCN(64,128,'relu')
    self.gcn3 = GCN(128,256,'relu')
    self.gcn4 = GCN(256,512)

    self.ip = IP()
  
  def encoder(self,X, A):
      # x = self.gcn1(X,A)
      # x = self.gcn2(x,A)

      x = self.gcn1(X,A)
      x = self.gcn2(x,A)
      x = self.gcn3(x,A)
      x = self.gcn4(x,A)
      return x
  
  def decoder(self,z):
    x = self.ip(z)
    return x
  
  def set_weights(self,weights):
    self.gcn1.weights = weights[0]
    self.gcn2.weights = weights[1]
    self.ip.weights = weights[4]
  
  def __call__(self,X,A):
    x = self.encoder(X,A)
    x = self.decoder(x)
    return x