from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import tanh
import tensorflow as tf
from tensorflow.keras import initializers
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import copy
from functools import partial


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


class WeightedCrossEntropyLogitsMetric(tf.keras.metrics.Metric):

  def __init__(self, w_p):
    super(WeightedCrossEntropyLogitsMetric, self).__init__(name='weighted_cross_entropy_with_logits')
    self.__loss = partial(tf.nn.weighted_cross_entropy_with_logits,pos_weight=w_p)
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.__losssum.assign_add(tf.reduce_sum(self.__loss(y_true, y_pred)))

  def result(self):
    return self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)

class L2MeanSquaredError(tf.keras.metrics.Metric):
  def __init__(self):
    super(L2MeanSquaredError, self).__init__(name='l2mse')
    self.__loss = tf.keras.metrics.MeanSquaredError()
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')

  def update_state(self, y_true, y_pred):
    l2loss = self.__loss(y_true, y_pred)**2
    self.__losssum.assign_add(tf.reduce_sum(l2loss))

  def result(self):
    return 1/2*self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)

class L2CategoricalCrossentropy(tf.keras.metrics.Metric):
  def __init__(self):
    super(L2CategoricalCrossentropy, self).__init__(name='l2cce')
    self.__loss = tf.keras.metrics.CategoricalCrossentropy()
    self.__losssum = self.add_weight(name='losssum', initializer='zeros')

  def update_state(self, y_true, y_pred):
    l2loss = self.__loss(y_true, y_pred)**2
    self.__losssum.assign_add(tf.reduce_sum(l2loss))

  def result(self):
    return 1/2*self.__losssum

  def reset_states(self):
    self.__losssum.assign(0)

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
  def __init__(self, early_stop_vars=None, save_weights_obj=None, learning_rate=1e-4, amsgrad=True):
    self.__optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=amsgrad)
    

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
        self.__optimizer.apply_gradients(zip(grads, self.trainable_variables))

      print ('Epoch {} finished'.format(self.__epochs_cnt))
      if history_learning_process:
        self.set_score_mode(False)
        train_cost_mtr = self.cost_mtr.metric_dataset(dataset)
        self.set_score_mode(True)
        train_score_mtr = self.score_mtr.metric_dataset(dataset)
        
        self.history['train_cost'].append(
            train_cost_mtr
        )
        self.history['train_score'].append(
            train_score_mtr
        )
        print('train_cost: {}, train_score: {}'.format(train_cost_mtr, train_score_mtr))

        if dataset_val is not None:
          # this is super hard code
          # kouleison
          self.set_score_mode(False)
          val_cost_mtr = self.cost_mtr.metric_dataset(dataset_val)
          self.set_score_mode(True)
          val_score_mtr = self.score_mtr.metric_dataset(dataset_val)

          self.history['val_cost'].append(
             val_cost_mtr
          )
          self.history['val_score'].append(
              val_score_mtr
          )
          print('val_cost: {}, val_score: {}'.format(val_cost_mtr, val_score_mtr))
        if dataset_test is not None:
          self.set_score_mode(False)
          test_cost_mtr = self.cost_mtr.metric_dataset(dataset_test)
          self.set_score_mode(True)
          test_score_mtr = self.score_mtr.metric_dataset(dataset_test)

          self.history['test_cost'].append(
            test_cost_mtr
          )
          self.history['test_score'].append(
            test_score_mtr
          )
          print('test_cost: {}, test_score: {}'.format(test_cost_mtr, test_score_mtr))

        if dataset_val is not None and dataset_test is not None:
          harmonic_score = 2*test_score_mtr*val_score_mtr/(test_score_mtr+val_score_mtr)

          self.history['harmonic_score'].append(
            harmonic_score
          )
          print('harmonic score: {}'.format(harmonic_score))

        if self.__early_stop is not None:
          if self.__early_stop.check_stop(copy.deepcopy(self.history)):
            print('Stop Training')
            break


class Knn(object):
  def __init__(self):
    self.__knn = KNeighborsClassifier(1)
  
  def fit(self, x, y):
    self.__knn.fit(x, y)

  def predict(self, x):
    return self.__knn.predict(x)
  
  def predict_dataset(self, dataset):
    pred = []
    for x, _, _, _ in dataset:
      pred.append(self.predict(x))
    pred = tf.concat(pred, axis=0)
    return pred
    

class GCN(tf.Module):
  def __init__(self,in_features,out_features, activation=None, name=None):
    if name is None:
      name = "gcn"
    super(GCN,self).__init__(name=name)
  
    self.weights = tf.Variable(
      tf.keras.initializers.GlorotUniform()(shape=[in_features, out_features]),
      name='weights'
    )
    self.bais = tf.Variable(
      tf.keras.initializers.GlorotUniform()(shape=[out_features]),
      name='bais'
    )
    self.activation = tf.keras.activations.get(activation)

  def __call__(self, X, Atld):
    x = tf.matmul(Atld,X)
    x = tf.matmul(x,self.weights) + self.bais
    x = self.activation(x)
    return x

class IP(tf.Module):
  def __init__(self,in_features=None,activation=None):
    super(IP,self).__init__(name='ip')
    self.weights = None
    if in_features is not None:
      self.weights = tf.Variable(
        tf.random.normal([in_features, in_features]), name='weights')

    self.activation = tf.keras.activations.get(activation)

  def __call__(self,inputs):
    x = inputs
    if self.weights is not None:
      x = tf.matmul(x,self.weights)
    x = tf.matmul(x,inputs,transpose_b=True)
    x = self.activation(x)
    return x

class Rnn(tf.keras.Model, Trainer):

  def __init__(self, embsD, knn, early_stop_vars=None, weights_outfile=None, learning_rate=1e-4, seed=None):
    tf.keras.Model.__init__(self,name='rnn')
    save_weights_obj = None
    if weights_outfile is not None:
      save_weights_obj = partial(self.save_weights, "../weights/weights_best_{}.tf".format(weights_outfile))
    Trainer.__init__(self, early_stop_vars, save_weights_obj, learning_rate)
    
    self.__knn = knn
    self.__score_mode = False

    self.__status = [
      [0],
      [1],
      [2],
      [1,2]
    ]

    self.score_mtr = MetricBase(self,
      [tf.keras.metrics.CategoricalAccuracy()],
      self.__status,
      [0, 0]
    )

    self.cost_mtr = MetricBase(self,
      [
        tf.keras.metrics.MeanSquaredError(),
        tf.keras.metrics.CategoricalCrossentropy()
      ],
      self.__status,
      [0,1],
      1
    )

    self.cost_loss = LossBase(self,
      [
        tf.keras.losses.MeanSquaredError(),
        tf.keras.losses.CategoricalCrossentropy()
      ],
      self.__status,
      [0, 1]
    )
    
    tf.random.set_seed(seed)
    
    x_dim = (24, 4)
    self.Input = Input(x_dim)

    self.lstm1 =  LSTM(24,return_sequences=True,name="lstm1")
    self.bidirectional1 = Bidirectional(self.lstm1,name="bidirectional1")

    self.lstm2 = LSTM(24,return_sequences=True,name="lstm2")
    self.bidirectional2 =  Bidirectional(self.lstm2,name="bidirectional2")

    self.lstm31 = LSTM(24,name="lstm31")
    self.bidirectional31 = Bidirectional(self.lstm31,name="bidirectional31")

    self.lstm32 = LSTM(24,name="lstm32")
    self.bidirectional32 = Bidirectional(self.lstm32,name="bidirectional32")

    self.denseout1 = Dense(embsD,activation=None,name="denseout1")
    self.denseout2 = Dense(2,activation="softmax",name="denseout2")

  def set_score_mode(self, score_mode):
    self.__score_mode = score_mode

  def call(self, inputs):

    x = self.bidirectional1(inputs[0])

    x = self.bidirectional2(x)

    x31 = self.bidirectional31(x)
    x32 = self.bidirectional32(x)

    y1 = self.denseout1(x31)

    # super hard code
    if self.__score_mode:
      y1 = self.__knn.predict(y1)

    y2 = self.denseout2(x32)

    return y1, y2

# 0 input
# 1 output | loss
# 2 output | mtr
class miniGAEv1(tf.Module, Trainer):
  def __init__(self, ft_number, w_p, early_stop_vars=None, weights_outfile=None, learning_rate=1e-2):
    tf.Module.__init__(self, name='gae')
    save_weights_obj = None
    if weights_outfile is not None:
      save_weights_obj = partial(self.save_weights, "../weights/weights_best_{}.tf".format(weights_outfile))
    Trainer.__init__(self, early_stop_vars, save_weights_obj, learning_rate)
    self.__score_mode = False

    self.__status = [
      [0],
      [0],
      [1,2]
    ]

    self.cost_mtr = MetricBase(self,
      [WeightedCrossEntropyLogitsMetric(w_p)],
      self.__status,
      [0],
      1
    )

    self.score_mtr = MetricBase(self,
      [tf.keras.metrics.BinaryAccuracy()],
      self.__status,
      [0]
    )

    self.cost_loss = LossBase(self,
      [partial(tf.nn.weighted_cross_entropy_with_logits,pos_weight=w_p)],
      self.__status,
      [0]
    )

    self.gcn1 = GCN(ft_number,16, "relu", "gcn1")
    self.gcn2 = GCN(16,32, name="gcn2")
    self.ip = IP()
  
  def encoder(self,X, Atld):
      x = self.gcn1(X,Atld)
      x = self.gcn2(x,Atld)
      return x
  
  def decoder(self,z):
    x = self.ip(z)
    return x
  
  def __call__(self,inputs):
    x = self.encoder(inputs[0],inputs[1])
    y = self.decoder(x)

    if self.__score_mode:
      y = tf.nn.sigmoid(y)

    return tuple((y,))
  
  # super hard code
  def set_score_mode(self, score_mode):
    self.__score_mode = score_mode

#-------------------------------#
#- significance NN -------------#
#-------------------------------#
class SGCN(tf.Module):
  def __init__(self, in_features,out_features, activation=None, firstlayer=False, name=None):
    super(SGCN,self).__init__(name=name)
  
    self.weights = tf.Variable(
      tf.keras.initializers.GlorotUniform()(shape=[in_features, out_features]),
      name='weights'
    )
    self.activation = tf.keras.activations.get(activation)

    self.__firstlayer = firstlayer
  def __call__(self, X, Atld):

    def phi(l):
      lsum2 = tf.reduce_sum(l,axis=2)
      lsum2 = tf.stack(l.shape[2]*[lsum2],axis=2)
      ch = (tf.sign(tf.abs(lsum2) - 10**(-5))+1)/2
      expl = tf.exp(l)
      expl_sum = tf.reduce_sum(expl,axis=2)
      expl_sum = tf.stack(l.shape[2]*[expl_sum],axis=2)
      out = self.activation(lsum2)
      out = out*(l/lsum2*ch + (1-ch)*(expl/expl_sum))
      return out

    l = 0
    if not self.__firstlayer:
      I = myeye(Atld.shape[0])
      F = tf.tensordot(I,Atld,[[1],[0]])
      F = tf.tensordot(F,X,[[1, 3],[0,1]])
      l = tf.matmul(F, self.weights) 
    else:
      l = tf.matmul(X, self.weights)
    
    out = phi(l)
    return out


# its good to create BaseModel ?!
class SGAE(tf.Module, Trainer):
  def __init__(self, list_f, w_p, early_stop_vars=None, weights_outfile=None, learning_rate=1e-2):
    tf.Module.__init__(self,  name="sgae")
    save_weights_obj = None
    if weights_outfile is not None:
      save_weights_obj = partial(self.save_weights, "../weights/weights_best_{}.tf".format(weights_outfile))
    Trainer.__init__(self, early_stop_vars, save_weights_obj, learning_rate)

    self.__score_mode = False

    self.__status = [
      [0],
      [0],
      [1,2]
    ]

    self.cost_mtr = MetricBase(self,
      [WeightedCrossEntropyLogitsMetric(w_p)],
      self.__status,
      [0],
      1
    )

    self.score_mtr = MetricBase(self,
      [tf.keras.metrics.BinaryAccuracy()],
      self.__status,
      [0]
    )

    self.cost_loss = LossBase(self,
      [partial(tf.nn.weighted_cross_entropy_with_logits,pos_weight=w_p)],
      self.__status,
      [0]
    )

    self.__num_stack = len(list_f)
    self.sgcn0 = SGCN(list_f[0], list_f[1], 'relu', True)
    for i in range(1, self.__num_stack-1):
      setattr(
        self, 'sgcn{}'.format(str(i)),
        SGCN(
          list_f[i],
          list_f[i+1],
          'relu', name='sgcn{}'.format(str(i))
        ))
    self.ip = IP()

  def set_score_mode(self, score_mode):
    self.__score_mode = score_mode

  def __call__(self, inputs):
    x = self.sgcn0(inputs[0], inputs[1])
    for i in range(1, self.__num_stack-1):
      x = getattr(self, 'sgcn{}'.format(str(i)))(x, inputs[1])
    
    x = tf.reduce_sum(x, axis=2)
    y = self.ip(x)

    if self.__score_mode:
       y = tf.nn.sigmoid(y)
    
    return tuple((y,))


#-------------------------------#
#- Extended NN -----------------#
#-------------------------------#

class SGAEext(tf.Module):
  def __init__(self, list_f):
    super(SGAEext,self).__init__(name='sgae_ext')
    self.__num_stack = len(list_f)
    self.sgcn0 = SGCN(list_f[0], list_f[1], 'relu', True)
    for i in range(1, self.__num_stack-1):
      setattr(
        self, 'sgcn{}'.format(str(i)),
        SGCN(
          list_f[i],
          list_f[i+1],
          'relu', name='sgcn{}'.format(str(i))
        ))
    self.ip = IP()


  def __call__(self, Xpr, Atld):
    ys = []
    x = self.sgcn0(Xpr, Atld)
    ys.append(x)
    for i in range(1, self.__num_stack-1):
      x = getattr(self, 'sgcn{}'.format(str(i)))(x, Atld)
      ys.append(x)
    
    x = tf.reduce_sum(x, axis=2)
    x = self.ip(x)
    ys.append(x)
    return ys

class RNNext(tf.keras.Model):
  def __init__(self, embsD):
    super(RNNext,self).__init__(name='rnnext')

    x_dim = (24, 4)
    self.Input = Input(x_dim)

    self.lstm1 =  LSTM(24,return_sequences=True,name="lstm1")
    self.bidirectional1 = Bidirectional(self.lstm1,name="bidirectional1")

    self.lstm2 = LSTM(24,return_sequences=True,name="lstm2")
    self.bidirectional2 =  Bidirectional(self.lstm2,name="bidirectional2")

    self.lstm31 = LSTM(24,name="lstm31")
    self.bidirectional31 = Bidirectional(self.lstm31,name="bidirectional31")

    self.lstm32 = LSTM(24,name="lstm32")
    self.bidirectional32 = Bidirectional(self.lstm32,name="bidirectional32")

    self.denseout1 = Dense(embsD,activation=None,name="denseout1")
    self.denseout2 = Dense(2,activation="softmax",name="denseout2")

  def call(self, udp_input):
    x = self.bidirectional1(udp_input)

    x = self.bidirectional2(x)

    x31 = self.bidirectional31(x)
    x32 = self.bidirectional32(x)

    y1 = self.denseout1(x31)
    y2 = self.denseout2(x32)

    return y1, y2
  


class ExtendedNN(tf.Module, Trainer):
  def __init__(self, embsD, list_f, w_p, knn, early_stop_vars=None, weights_outfile=None, learning_rate=1e-4):
    tf.Module.__init__(self,  name="extnn")
    save_weights_obj = None
    if weights_outfile is not None:
      save_weights_obj = partial(self.save_weights, "./weights/weights_best_{}.tf".format(weights_outfile))
    Trainer.__init__(self, early_stop_vars, save_weights_obj, learning_rate)
    self.__knn = knn
    self.__score_mode = False

    self.__status = [
      [0],
      [0],
      [0],
      [1,2],
      [1],
      [2],
      [1,2],
    ]

    self.score_mtr = MetricBase(self,[
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.CategoricalAccuracy()],
      self.__status,
      [0, 1, 1]
    )

    self.cost_mtr = MetricBase(self,[
        WeightedCrossEntropyLogitsMetric(w_p),
        L2MeanSquaredError(),
        L2CategoricalCrossentropy()
        # tf.keras.metrics.MeanSquaredError(),
        # tf.keras.metrics.CategoricalCrossentropy()
      ],
      self.__status,
      [0,1,2],
      1
    )

    self.cost_loss = LossBase(self,
      [
        partial(tf.nn.weighted_cross_entropy_with_logits,pos_weight=w_p),
        lambda ytr, ypr: 1/2*tf.keras.losses.MeanSquaredError()(ytr, ypr)**2,
        lambda ytr, ypr: 1/2*tf.keras.losses.CategoricalCrossentropy()(ytr, ypr)**2,
        # tf.keras.losses.MeanSquaredError(),
        # tf.keras.losses.CategoricalCrossentropy(),
      ],
      self.__status,
      [0, 1, 2]
    )
    self.__num_stack = len(list_f)

    self.sgae = SGAEext(list_f)
    self.rnn0 = RNNext(embsD)
    for i in range(1, self.__num_stack-1):
      setattr(self, 'rnn{}'.format(str(i)),RNNext(embsD))
  
  
  def significance(self, x):
      N = x.shape[1]-1
      I = tf.reshape(tf.eye(N,N),(1,N,N))
      I = tf.concat([I, tf.zeros((N,N,N))],axis=0)
      I = tf.concat([tf.zeros((N+1,1,N)), I],axis=1)
      s = tf.reduce_sum(x,axis=3)
      s = tf.tensordot(s,I,[[1,2],[0,1]])
      s = tf.nn.relu(1-tf.pow(10,-s))
      return s

  def set_score_mode(self, score_mode):
    self.__score_mode = score_mode

  
  def __call__(self, inputs):

    sgcnouts = self.sgae(inputs[0], inputs[1])
    outs = []
    for i, out in  enumerate(sgcnouts[:-1]):
      s = self.significance(out)
      sx = inputs[2]*tf.stack(24*[s],axis=1)
      rhat, ahat = getattr(self, 'rnn{}'.format(str(i)))(sx)
      if self.__score_mode:
        rhat = self.__knn.predict(rhat)
      outs += [rhat, ahat]
    
    if self.__score_mode:
      sgcnouts[-1]= tf.nn.sigmoid(sgcnouts[-1])

    return [sgcnouts[-1]]+outs
    





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

def myeye(N):
    return tf.cast([[[1 if i==j and j==w  else 0 for i in range(N)] for j in range(N)] for w in range(N)], tf.float32)

def Pr0(Atld, X):
  I0 = myeye(Atld.shape[0])
  I2 = myeye(Atld.shape[2])
  Xpr = tf.tensordot(I0, Atld, [[1],[0]])
  Xpr = tf.tensordot(Xpr, I2, [[3],[1]])
  Xpr = tf.tensordot(Xpr, X, [[1,4],[0,1]])
  return Xpr


  
  