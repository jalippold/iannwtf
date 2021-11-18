import tensorflow as tf
from MyCustomWineLayer import MyDenseLayer

# this class represents a model which consists of
# 2 hidden layers and 1 output layer


class MyModel(tf.keras.Model):

    def __init__(self, kernel_reg=None, bias_reg=None, dropout=False, dropout_rate= 0.2):
        super(MyModel, self).__init__()
        #first hidden layer with 11 inputs and 64 neurons
        self.dense1 = MyDenseLayer(32,kernel_regularizer=kernel_reg,bias_regularizer=bias_reg)
        # second hidden layer with 64 inputs and 64 neurons
        self.dense2 = MyDenseLayer(32,kernel_regularizer=kernel_reg,bias_regularizer=bias_reg)
        # output layer layer with 64 inputs and 1 neurons
        self.out = MyDenseLayer(1,kernel_regularizer=kernel_reg,bias_regularizer=bias_reg)

        # experimenting with dropout like in:
        # https://towardsdatascience.com/understanding-and-implementing-dropout-in-tensorflow-and-keras-a8a3a02c1bfa
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate, seed=42)

    def call(self, inputs, training=None):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        x = self.dense1(inputs)
        if self.dropout:
            x = self.dropout_layer(x, training=training)
        x = self.dense2(x)
        if self.dropout:
            x = self.dropout_layer(x, training=training)
        x = self.out(x)
        return x