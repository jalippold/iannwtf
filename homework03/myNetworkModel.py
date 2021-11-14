import tensorflow as tf
from CustomLayer import MyDenseLayer
from MyCustomOutputLayer import MyDenseOutputLayer

# this class represents a model which consists of
# 2 hidden layers and 1 output layer


class MyModel(tf.keras.Model):

    def __init__(self, num_inputs):
        super(MyModel, self).__init__()
        #first hidden layer with 1000 inputs and 256 neurons
        self.dense1 = MyDenseLayer(256)
        # second hidden layer with 256 inputs and 256 neurons
        self.dense2 = MyDenseLayer(256)
        # output layer layer with 256 inputs and 10 neurons
        self.out = MyDenseOutputLayer(10)

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
