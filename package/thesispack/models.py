from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow.keras import Model
from thesispack.base import BaseNeuralNetwork, MetricBase, LossBase
from thesispack.metrics import WeightedCrossEntropyWithLogits as mWCEL, MeanSquaredErrorWithLambda as mMSEL, CategoricalCrossentropyWithLambda as mCCEL, MeanEuclidianError
from thesispack.layers import GCN, SGCN, IP, Ap_LSTM, Attention, FC
from thesispack.losses import WeightedCrossEntropyWithLogits as lWCEL, MeanSquaredErrorWithLambda as lMSEL, CategoricalCrossentropyWithLambda as lCCEL
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional



class miniGAE(tf.Module, BaseNeuralNetwork):
    def __init__(self, ft_number, w_p, norm=1, early_stop_vars=None, weights_outfile=None, optimizer="SGD",
                 learning_rate=1e-2):
        tf.Module.__init__(self, name='gae')
        status = [
            [0],
            [0],
            [1, 2]
        ]
        BaseNeuralNetwork.__init__(self, status, early_stop_vars, weights_outfile, optimizer, learning_rate)

        self.cost_mtr = MetricBase(self,
                                   [mWCEL(w_p, norm)],
                                   status,
                                   [0],
                                   1
                                   )

        self.score_mtr = MetricBase(self,
                                    [tf.keras.metrics.BinaryAccuracy()],
                                    status,
                                    [0]
                                    )

        self.cost_loss = LossBase(self,
                                  [lambda ytr, ypr: norm * tf.reduce_mean(
                                      tf.nn.weighted_cross_entropy_with_logits(ytr, ypr, w_p))],
                                  status,
                                  [0]
                                  )

        self.gcn1 = GCN(ft_number, 16, "relu", "gcn1")
        self.gcn2 = GCN(16, 32, name="gcn2")
        self.ip = IP()

    def encoder(self, X, Atld):
        x = self.gcn1(X, Atld)
        x = self.gcn2(x, Atld)
        return x

    def decoder(self, z):
        x = self.ip(z)
        return x

    def __call__(self, inputs):
        x = self.encoder(inputs[0], inputs[1])
        y = self.decoder(x)

        if self.get_score_mode():
            y = tf.nn.sigmoid(y)

        return tuple((y,))


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


class GCNforDepth(tf.Module, BaseNeuralNetwork):
    def __init__(self, nf0, nc, depth=1, nfi=64):
        tf.Module.__init__(self, name='my_gcn')
        status = [
            [0],
            [0],
            [1, 2]
        ]
        BaseNeuralNetwork.__init__(self, status, learning_rate=1e-2)
        self.depth = depth

        self.score_mtr = MetricBase(self,
                                    [tf.keras.metrics.CategoricalAccuracy()],
                                    status,
                                    [0]
                                    )
        self.cost_mtr = MetricBase(self,
                                   [tf.keras.metrics.CategoricalCrossentropy()],
                                   status,
                                   [0],
                                   1
                                   )
        self.cost_loss = LossBase(self,
                                  # make custom cost
                                  [tf.keras.losses.CategoricalCrossentropy()],
                                  status,
                                  [0]
                                  )
        depthi = '1'
        setattr(self, 'gcn{}'.format(depthi), GCN(nf0, nfi, 'relu'))
        if self.depth > 1:
            l = 0.3 / (self.depth - 1)
        else:
            l = 1
        for d in range(1, self.depth):
            depthi = str(d + 1)
            setattr(self, 'gcn{}'.format(depthi), GCN(nfi, nfi, 'relu'))
            drop = l * d + 0.1

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = FC(nf0 * nfi, 256, 'relu')
        self.out = FC(256, nc, 'softmax')

    def __call__(self, inputs):
        depthi = '1'
        x = getattr(self, 'gcn{}'.format(depthi))(inputs[0], inputs[1])
        for d in range(1, self.depth):
            depthi = str(d + 1)
            x = getattr(self, 'gcn{}'.format(depthi))(x, inputs[1])

        x = self.flatten(x)
        x = self.fc1(x)
        y = self.out(x)
        y = tuple([y])
        return y

    def set_knn_out(self, knn_out):
        pass


class Rnn(Model, BaseNeuralNetwork):

    def __init__(self, embsD, knn, early_stop_vars=None, weights_outfile=None, optimizer="SGD", learning_rate=1e-4):
        Model.__init__(self, name='rnn')
        status = [
            [0],
            [1],
            [2],
            [1, 2]
        ]
        BaseNeuralNetwork.__init__(self, status, early_stop_vars, weights_outfile, optimizer, learning_rate)

        self.knn = knn
        self.score_mtr = MetricBase(self,
                                    [tf.keras.metrics.CategoricalAccuracy()],
                                    status,
                                    [0, 0]
                                    )

        self.cost_mtr = MetricBase(self,
                                   [
                                       tf.keras.metrics.MeanSquaredError(),
                                       tf.keras.metrics.CategoricalCrossentropy()
                                   ],
                                   status,
                                   [0, 1],
                                   1
                                   )

        self.cost_loss = LossBase(self,
                                  [
                                      tf.keras.losses.MeanSquaredError(),
                                      tf.keras.losses.CategoricalCrossentropy()
                                  ],
                                  status,
                                  [0, 1]
                                  )

        x_dim = (24, 4)
        self.Input = Input(x_dim)

        self.lstm1 = LSTM(24, batch_input_shape=(128, 24, 4), return_sequences=True, name="lstm1")
        self.bidirectional1 = Bidirectional(self.lstm1, name="bidirectional1")

        self.lstm2 = LSTM(24, return_sequences=True, name="lstm2")
        self.bidirectional2 = Bidirectional(self.lstm2, name="bidirectional2")

        self.lstm31 = LSTM(24, name="lstm31")
        self.bidirectional31 = Bidirectional(self.lstm31, name="bidirectional31")

        self.lstm32 = LSTM(24, name="lstm32")
        self.bidirectional32 = Bidirectional(self.lstm32, name="bidirectional32")

        self.denseout1 = Dense(embsD, activation=None, name="denseout1")
        self.denseout2 = Dense(2, activation="softmax", name="denseout2")

    def call(self, inputs):
        x = self.bidirectional1(inputs[0])

        x = self.bidirectional2(x)

        x31 = self.bidirectional31(x)
        x32 = self.bidirectional32(x)

        y1 = self.denseout1(x31)

        # super hard code
        if self.get_score_mode():
            y1 = self.knn.predict(y1)

        y2 = self.denseout2(x32)

        return y1, y2


class SGAE(tf.Module):
    def __init__(self, list_f, Arows=None):
        super(SGAE, self).__init__(name='sgae')
        self.__num_stack = len(list_f)
        self.sgcn0 = SGCN(list_f[0], list_f[1], 'relu', True)
        for i in range(1, self.__num_stack - 1):
            setattr(
                self, 'sgcn{}'.format(str(i)),
                SGCN(
                    list_f[i],
                    list_f[i + 1],
                    'relu', name='sgcn{}'.format(str(i))
                ))
        if Arows is not None:
            self.ip = IP(list_f[-1])
        else:
            self.ip = IP()

    def __call__(self, Xpr, Atld):
        ys = []
        x = self.sgcn0(Xpr, Atld)
        ys.append(x)
        for i in range(1, self.__num_stack - 1):
            x = getattr(self, 'sgcn{}'.format(str(i)))(x, Atld)
            ys.append(x)

        x = tf.reduce_sum(x, axis=2)
        x = self.ip(x)
        ys.append(x)
        return ys


class ExtRNN(tf.keras.Model):
  def __init__(self, embsD):
    super(ExtRNN,self).__init__(name='ext_rnn')

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


class ExtendedNN(tf.Module, BaseNeuralNetwork):
    def __init__(self, embsD, list_f, w_p, norm, knn, early_stop_vars=None, weights_outfile=None, optimizer="SGD",
                 learning_rate=1e-1, lamda=1):
        tf.Module.__init__(self, name="extnn")
        status = [
            [0],
            [0],
            [0],
            [1, 2],
            [1],
            [2],
            [1, 2]
        ]
        BaseNeuralNetwork.__init__(self, status, early_stop_vars, weights_outfile, optimizer, learning_rate)
        self.knn = knn

        self.score_mtr = MetricBase(self, [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.CategoricalAccuracy()],
                                    status,
                                    [0, 1, 1]
                                    )

        self.cost_mtr = MetricBase(self, [
            mWCEL(w_p, norm),
            mMSEL(lamda),
            mCCEL(lamda)
        ],
                                   status,
                                   [0, 1, 2],
                                   1
                                   )
        self.cost_loss = LossBase(self,
                                  [
                                      lWCEL(w_p, norm),
                                      lMSEL(lamda),
                                      lCCEL(lamda)
                                  ],
                                  status,
                                  [0, 1, 2]
                                  )
        self.__num_stack = len(list_f)

        self.sgae = SGAE(list_f)
        self.rnn0 = ExtRNN(embsD)
        for i in range(1, self.__num_stack - 1):
            setattr(self, 'rnn{}'.format(str(i)), ExtRNN(embsD))

    def significance(self, x):
        N = x.shape[1] - 1
        I = tf.reshape(tf.eye(N, N), (1, N, N))
        I = tf.concat([I, tf.zeros((N, N, N))], axis=0)
        I = tf.concat([tf.zeros((N + 1, 1, N)), I], axis=1)
        s = tf.reduce_sum(x, axis=3)
        s = tf.tensordot(s, I, [[1, 2], [0, 1]])
        s = tf.nn.relu(1 - tf.pow(10, -s))
        return s


    def __call__(self, inputs):

        sgcnouts = self.sgae(inputs[0], inputs[1])
        outs = []
        for i, out in enumerate(sgcnouts[:-1]):
            s = self.significance(out)
            sx = inputs[2] * tf.stack(24 * [s], axis=1)
            rhat, ahat = getattr(self, 'rnn{}'.format(str(i)))(sx)
            if self.get_score_mode():
                rhat = self.knn.predict(rhat)
            outs += [rhat, ahat]

        if self.get_score_mode():
            sgcnouts[-1] = tf.nn.sigmoid(sgcnouts[-1])

        return [sgcnouts[-1]] + outs


class RttRnnAt(Model, BaseNeuralNetwork):
    def __init__(self, early_stop_vars=None, min_xy=0, max_xy=1):
        Model.__init__(self, name="rtt_rnn")
        status = [
            [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
            [1, 2]
        ]
        BaseNeuralNetwork.__init__(self, status, early_stop_vars, None, "Adam", 1e-4)
        self.score_mtr = MetricBase(self,
                                    [MeanEuclidianError(min_xy, max_xy)],
                                    status,
                                    [0]
                                    )

        self.cost_mtr = MetricBase(self,
                                   [tf.keras.metrics.MeanSquaredError()],
                                   status,
                                   [0],
                                   1
                                   )

        self.cost_loss = LossBase(self,
                                  [tf.keras.losses.MeanSquaredError()],
                                  status,
                                  [0]
                                  )

        units = 256
        self.ap_ant1_lstm = Ap_LSTM(units, return_sequences=True, batch_input_shape=(128, 12, 114))
        self.ap_ant1_at = Attention(units)

        self.ap_ant2_lstm = Ap_LSTM(units, return_sequences=True, batch_input_shape=(128, 12, 114))
        self.ap_ant2_at = Attention(units)

        self.tod_lstm = LSTM(units, return_sequences=True, name='tod_ap1_lstm', batch_input_shape=(128, 7, 12))

        self.tod_at1_lstm = LSTM(units, return_sequences=False, name='tod_ap1_lstm')

        self.tod_at2_lstm = LSTM(units, return_sequences=False, name='tod_ap2_lstm')

        self.denseout = Dense(2, activation=None, name="denseout")

    def call(self, inputs):
        xap_ant1 = self.ap_ant1_lstm(inputs[1:13])
        xap_ant1_at = self.ap_ant2_at(xap_ant1)

        xap_ant2 = self.ap_ant1_lstm(inputs[13:])
        xap_ant2_at = self.ap_ant2_at(xap_ant2)

        x = self.tod_lstm(inputs[0])

        xap_ant1_tod = tf.concat([tf.expand_dims(xap_ant1_at, 1), x], axis=-2)
        xap_ant2_tod = tf.concat([tf.expand_dims(xap_ant2_at, 1), x], axis=-2)

        xap_ant1_tod = self.tod_at1_lstm(xap_ant1_tod)

        xap_ant2_tod = self.tod_at2_lstm(xap_ant2_tod)

        y = tf.concat([xap_ant1_tod, xap_ant2_tod], axis=-1)
        y = self.denseout(y)
        return tuple((y,))


# -------------------------------#
# - Pre codes -------------------#
# -------------------------------#
class GAE(tf.Module):
    def __init__(self, ft_number):
        super(GAE, self).__init__(name='gae')

        # self.gcn1 = GCN(ft_number,128,'relu')
        # self.gcn2 = GCN(128,256)

        self.gcn1 = GCN(ft_number, 64, 'relu')
        self.gcn2 = GCN(64, 128, 'relu')
        self.gcn3 = GCN(128, 256, 'relu')
        self.gcn4 = GCN(256, 512)

        self.ip = IP()

    def encoder(self, X, A):
        # x = self.gcn1(X,A)
        # x = self.gcn2(x,A)

        x = self.gcn1(X, A)
        x = self.gcn2(x, A)
        x = self.gcn3(x, A)
        x = self.gcn4(x, A)
        return x

    def decoder(self, z):
        x = self.ip(z)
        return x

    def set_weights(self, weights):
        self.gcn1.weights = weights[0]
        self.gcn2.weights = weights[1]
        self.ip.weights = weights[4]

    def __call__(self, X, A):
        x = self.encoder(X, A)
        x = self.decoder(x)
        return x