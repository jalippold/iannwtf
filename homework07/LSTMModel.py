import tensorflow as tf
from LSTMWrapper import LSTMWrapper
from LSTMCell import CustomLSTMCell
class LSTM_Model(tf.keras.Model):
    def __init__(self, units):
        super(LSTM_Model, self).__init__()
        self.LSTMWrapper = LSTMWrapper(CustomLSTMCell(units), return_sequences=False)
        self.out = tf.keras.layers.Dense(2, activation="softmax")

    # @tf.function(experimental_relax_shapes=True)
    def call(self, data):
        x = self.RNNWrapper(data)
        x = self.out(x)
        return x