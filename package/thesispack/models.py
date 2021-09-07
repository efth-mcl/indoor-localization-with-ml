from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from thesispack.base import BaseNeuralNetwork, MetricBase, LossBase
from thesispack.metrics import WeightedCrossEntropyWithLogits as mWCEL
from thesispack.layers import GCN, IP, FC


class MiniGAE(tf.Module, BaseNeuralNetwork):
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


class GCNwithDepth(tf.Module, BaseNeuralNetwork):
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