import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Bidirectional, Dense
from algomorphism.model.base import BaseNeuralNetwork, LossBase, MetricBase
from algomorphism.model.layers import IP
from algomorphism.model.losses import WeightedCrossEntropyWithLogits as lWCEL, MeanSquaredErrorWithLambda as lMSEL, CategoricalCrossEntropyWithLambda as lCCEL
from algomorphism.method.opt import three_d_identity_matrix
from algomorphism.model.metrics import WeightedCrossEntropyWithLogits as mWCEL, MeanSquaredErrorWithLambda as mMSEL, CategoricalCrossEntropyWithLambda as mCCEL


class RNNbase(tf.keras.Model):
    def __init__(self, embsD):
        super(RNNbase, self).__init__(name='rnnb')

        x_dim = (24, 4)
        self.Input = Input(x_dim)

        self.lstm1 = LSTM(256, return_sequences=True, name="lstm1")
        self.bidirectional1 = Bidirectional(self.lstm1, name="bidirectional1")

        self.lstm2 = LSTM(256, return_sequences=True, name="lstm2")
        self.bidirectional2 = Bidirectional(self.lstm2, name="bidirectional2")

        self.lstm31 = LSTM(512, name="lstm31")
        self.bidirectional31 = Bidirectional(self.lstm31, name="bidirectional31")

        self.lstm32 = LSTM(512, name="lstm32")
        self.bidirectional32 = Bidirectional(self.lstm32, name="bidirectional32")

        self.fc11 = Dense(512, activation='tanh', name="fc11")
        self.fc21 = Dense(1024, activation='tanh', name="fc21")
        self.fc31 = Dense(2048, activation='tanh', name="fc31")

        self.f12 = Dense(640, activation='relu', name="fc12")

        self.denseout1 = Dense(embsD, activation=None, name="denseout1")
        self.denseout2 = Dense(2, activation="softmax", name="denseout2")

    def call(self, x):
        x = self.bidirectional1(x)

        x = self.bidirectional2(x)

        x1 = self.bidirectional31(x)
        x2 = self.bidirectional32(x)

        x1 = self.fc11(x1)
        x1 = self.fc21(x1)
        x1 = self.fc31(x1)

        x2 = self.f12(x2)

        y1 = self.denseout1(x1)

        y2 = self.denseout2(x2)

        return y1, y2


class Rnn(tf.keras.Model, BaseNeuralNetwork):

    def __init__(self, dataset,  embsD, knn, early_stop_vars=None, weights_outfile=None, optimizer="SGD", learning_rate=1e-4):
        tf.keras.Model.__init__(self, name='rnn')
        status = [
            [0],
            [1],
            [2],
            [1, 2]
        ]

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

        BaseNeuralNetwork.__init__(self, status, dataset, early_stop_vars, weights_outfile, optimizer, learning_rate)

        self.rnnbase = RNNbase(embsD)

    def call(self, inputs, is_score=False):
        y1, y2 = self.rnnbase.call(inputs[0])

        if is_score:
            y1 = self.knn.predict(y1)

        return y1, y2


class SGCN(tf.Module):
    def __init__(self, in_features, out_features, activation=None, firstlayer=False, name=None):
        super(SGCN, self).__init__(name=name)

        self.weights = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[in_features, out_features]),
            name='weights'
        )
        self.activation = tf.keras.activations.get(activation)

        self.__firstlayer = firstlayer

    def __call__(self, X, Atld):

        def phi(l):
            lsum2 = tf.reduce_sum(l, axis=2)
            lsum2 = tf.stack(l.shape[2] * [lsum2], axis=2)
            ch = (tf.sign(tf.abs(lsum2) - 10 ** (-2)) + 1) / 2
            expl = tf.exp(l)
            expl_sum = tf.reduce_sum(expl, axis=2)
            expl_sum = tf.stack(l.shape[2] * [expl_sum], axis=2)
            out = self.activation(lsum2)
            out = out * (l / lsum2 * ch + (1 - ch) * (expl / expl_sum))
            return out

        if not self.__firstlayer:
            I = three_d_identity_matrix(Atld.shape[0])
            I = tf.cast(I, tf.float32)
            F = tf.tensordot(I, Atld, [[1], [0]])
            F = tf.tensordot(F, X, [[1, 3], [0, 1]])
            l = tf.matmul(F, self.weights)
        else:
            l = tf.matmul(X, self.weights)

        out = phi(l)
        return out

class SGAE(tf.Module):
    def __init__(self, list_f, ipw=False):
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
        if ipw:
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


class ExtendedNN(tf.Module, BaseNeuralNetwork):
    def __init__(self, dataset, embsD, list_f, w_p, norm, knn, early_stop_vars=None, weights_outfile=None, optimizer=None,
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
                mCCEL(lamda)],
           status,
               [0, 1, 2],
               1)

        self.cost_loss = LossBase(self,
                                  [
                                      lWCEL(w_p, norm),
                                      lMSEL(lamda),
                                      lCCEL(lamda)
                                  ],
                                  status,
                                  [0, 1, 2]
                                  )

        BaseNeuralNetwork.__init__(self, status, dataset, early_stop_vars, weights_outfile, optimizer, learning_rate)

        self.__num_stack = len(list_f)

        self.sgae = SGAE(list_f, ipw=True)
        self.rnn0 = RNNbase(embsD)
        for i in range(1, self.__num_stack):
            setattr(self, 'rnn{}'.format(str(i)), RNNbase(embsD))

    def significance(self, x):
        N = x.shape[1] - 1
        I = tf.reshape(tf.eye(N, N), (1, N, N))
        I = tf.concat([I, tf.zeros((N, N, N))], axis=0)
        I = tf.concat([tf.zeros((N + 1, 1, N)), I], axis=1)
        s = tf.reduce_sum(x, axis=3)
        s = tf.tensordot(s, I, [[1, 2], [0, 1]])
        s = tf.nn.relu(1 - tf.pow(10, -s))
        return s


    def __call__(self, inputs, is_score=False):
        sgcnouts = self.sgae(inputs[0], inputs[1])
        outs = []
        for i, out in enumerate(sgcnouts[:-1]):
            s = self.significance(out)
            sx = inputs[2] * tf.stack(24 * [s], axis=1)
            rhat, ahat = getattr(self, 'rnn{}'.format(str(i)))(sx)
            if is_score:
                rhat = self.knn.predict(rhat)
            outs += [rhat, ahat]

        if is_score:
            sgcnouts[-1] = tf.nn.sigmoid(sgcnouts[-1])

        return [sgcnouts[-1]] + outs


