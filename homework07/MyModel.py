import tensorflow as tf
from lstmlayer import LSTM_Layer
from lstmcell import LSTM_Cell


class LSTMModel(tf.keras.Model):
    def __init__(self, units, batchsize):
        super(LSTMModel, self).__init__()
        self.batchsize = batchsize
        self.lstmLayer = LSTM_Layer(LSTM_Cell(units))
        self.dense_for_out = tf.keras.layers.Dense(128, activation="relu")
        self.out = tf.keras.layers.Dense(2, activation="softmax")

    # @tf.function(experimental_relax_shapes=True)
    def call(self, data):
        first_states = self.lstmLayer.zero_states(self.batchsize)
        x = self.lstmLayer(x=data, states=first_states)
        x = self.dense_for_out(x)
        x = self.out(x)
        return x
