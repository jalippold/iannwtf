import tensorflow as tf
from lstmlayer import LSTM_Layer
from lstmcell import LSTM_Cell


class LSTMModel(tf.keras.Model):
    def __init__(self, units, batchsize):
        super(LSTMModel, self).__init__()
        self.batchsize = batchsize
        # use embedding
        self.embedding = tf.keras.layers.Dense(24, activation='sigmoid')
        self.lstmLayer = LSTM_Layer(LSTM_Cell(units))
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    # @tf.function(experimental_relax_shapes=True)
    def call(self, data):
        first_states = self.lstmLayer.zero_states(self.batchsize)
        x = self.embedding(data)
        x = self.lstmLayer(x=x, states=first_states)
        print('before')
        print(type(x))
        print(type(x[:][-1]))
        print(len(x[:][-1]))
        x = x[:][-1]
        x = tf.slice(x,[0,4,0],[-1,-1,-1])
        print(x)
        print('after')
        x = self.out(x)
        return x
