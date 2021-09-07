from thesispack.base import BaseNeuralNetwork, MetricBase, LossBase
from thesispack.layers import Attention
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense


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

class RttRnnAt(tf.keras.Model, BaseNeuralNetwork):
    def __init__(self, early_stop_vars=None, min_xy=0, max_xy=1):
        tf.keras.Model.__init__(self, name="rtt_rnn")
        status = [
            [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0],
            [1, 2]
        ]
        BaseNeuralNetwork.__init__(self, status, early_stop_vars, None, "Adam", 1e-4)

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