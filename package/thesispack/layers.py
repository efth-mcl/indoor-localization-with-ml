import tensorflow as tf
from tensorflow.python.keras.layers import LSTM, Dense
from thesispack.methods.nn import n_identity_matrix



#tf Modules
class FC(tf.Module):
    def __init__(self, in_features,out_features,activation=None):
        super(FC,self).__init__(name="fc")

        self.weights = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[in_features, out_features]),
            name='weights'
        )
        self.bais = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[out_features]),
            name='bais'
        )
        self.activation = tf.keras.activations.get(activation)

    def __call__(self,inputs):
        x = tf.matmul(inputs,self.weights) + self.bais
        x = self.activation(x)
        return x

class GCN(tf.Module):
    def __init__(self, in_features, out_features, activation=None, name=None):
        if name is None:
            name = "gcn"
        super(GCN, self).__init__(name=name)

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
        x = tf.matmul(Atld, X)
        x = tf.matmul(x, self.weights) + self.bais
        x = self.activation(x)
        return x


class IP(tf.Module):
    def __init__(self, in_features=None, activation=None):
        super(IP, self).__init__(name='ip')
        self.weights = None
        if in_features is not None:
            self.weights = tf.Variable(
                tf.random.normal([in_features, in_features]), name='weights')

        self.activation = tf.keras.activations.get(activation)

    def __call__(self, inputs):
        x = inputs
        if self.weights is not None:
            x = tf.matmul(x, self.weights)
        x = tf.matmul(x, inputs, transpose_b=True)
        x = self.activation(x)
        return x


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
            ch = (tf.sign(tf.abs(lsum2) - 10 ** (-5)) + 1) / 2
            expl = tf.exp(l)
            expl_sum = tf.reduce_sum(expl, axis=2)
            expl_sum = tf.stack(l.shape[2] * [expl_sum], axis=2)
            out = self.activation(lsum2)
            out = out * (l / lsum2 * ch + (1 - ch) * (expl / expl_sum))
            return out

        l = 0
        if not self.__firstlayer:
            I = n_identity_matrix(Atld.shape[0])
            F = tf.tensordot(I, Atld, [[1], [0]])
            F = tf.tensordot(F, X, [[1, 3], [0, 1]])
            l = tf.matmul(F, self.weights)
        else:
            l = tf.matmul(X, self.weights)

        out = phi(l)
        return out

# ref: tensorflow transformers
class Attention(tf.keras.layers.Layer):
    def __init__(self, units, comp_sum=False):
        super(Attention, self).__init__(name='attention')

        self.fc = Dense(units, activation='tanh')
        self.V = Dense(1)

    def call(self, deep_features):
        at = self.fc(deep_features)
        at = self.V(at)

        at = tf.nn.softmax(at, axis=1)

        at = tf.multiply(at, deep_features)

        at = tf.reduce_sum(at, axis=-2)

        return at


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, comp_sum=False):
        super(BahdanauAttention, self).__init__(name='bahdanau_attention')

        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, deep_features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        at = tf.nn.tanh(tf.add(self.W1(deep_features), self.W1(hidden)))
        at = self.V(at)

        at = tf.nn.softmax(at, axis=1)
        at = tf.multiply(at, deep_features)
        at = tf.reduce_sum(at, axis=-2)

        return at


class LSTM_At(tf.keras.layers.Layer):
    def __init__(self, units, name, batch_input_shape=None):
        super(LSTM_At, self).__init__(name=name)

        self.lstm1 = LSTM(units, return_sequences=True, batch_input_shape=batch_input_shape)
        self.at1 = Attention(units)

    def call(self, inpt, comp_sum=False):
        x = self.lstm1(inpt)
        y = self.at1(x)

        return y


class Ap_LSTM(tf.keras.layers.Layer):
    def __init__(self, units, return_sequences=False, batch_input_shape=None):

        super(Ap_LSTM, self).__init__(name='ap_lstm')
        for i in range(1, 13):
            setattr(
                self,
                'lstm_ap{}'.format(str(i)),
                LSTM(units, return_sequences=return_sequences, name='lstm_ap{}'.format(str(i)),
                     batch_input_shape=batch_input_shape)
            )

    def call(self, inputs):

        x = getattr(self, 'lstm_ap{}'.format(str(1)))(inputs[0])
        x = tf.expand_dims(x, 1)
        for i in range(1, 12):
            xi = getattr(self, 'lstm_ap{}'.format(str(i + 1)))(inputs[i])
            xi = tf.expand_dims(xi, 1)
            x = tf.concat([x, xi], axis=1)

        y = tf.reduce_sum(x, axis=1)
        return y