import tensorflow as tf
from MyCustomWineLayer import MyDenseLayer

# this class represents a model which consists of
# 2 hidden layers and 1 output layer


class MyModel(tf.keras.Model):

    def __init__(self, num_inputs):
        super(MyModel, self).__init__()
        #first hidden layer with 11 inputs and 64 neurons
        self.dense1 = MyDenseLayer(32,kernel_regularizer='l1',bias_regularizer='l1')
        # second hidden layer with 64 inputs and 64 neurons
        self.dense2 = MyDenseLayer(32,kernel_regularizer='l1',bias_regularizer='l1')
        # output layer layer with 64 inputs and 1 neurons
        self.out = MyDenseLayer(1,kernel_regularizer='l1',bias_regularizer='l1')

    def call(self, inputs):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x
