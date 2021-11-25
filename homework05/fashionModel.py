import tensorflow as tf
from MyDenseLayer import MyDenseLayer

class MyModel(tf.keras.Model):
    """
    This class represents a model which consists of a convolutional layer + one max pooling layer + one flattening layer 
    + one dense layer + one output layer
    """

    def __init__(self, kernel_reg=None, bias_reg=None):
        super(MyModel, self).__init__()

        #first hidden layer 2DConvoLayer
        self.convo = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same',
                                            activation='relu')
        # second hidden layer with pooling
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')

        # layer to flatten the tensors, has no effect on the batch size
        self.flatten = tf.keras.layers.Flatten()

        #third hidden layer; dense layer with 100 neurons
        self.dense1 = MyDenseLayer(100, activation='relu', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg) 
        
        # outputlayer
        self.out = MyDenseLayer(10, activation='softmax', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)

    @tf.function
    def call(self, inputs):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        x = self.convo(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.out(x)
        return x



class MyModelTwoConv(tf.keras.Model):
    """
    This model implements a convolutional layer followed by a max-pooling layer followed by a second convolutional layer
    followed by a flattening layer and finally followed by a custom dense layer for classification.
    """

    def __init__(self, kernel_reg=None, bias_reg=None):
        super(MyModelTwoConv, self).__init__()

        #first hidden layer 2DConvoLayer
        self.convo = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same',
                                            activation='relu')
        # second hidden layer with pooling
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')

        #third hidden layer 2DConvoLayer
        self.convo2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same',
                                            activation='relu')

        # layer to flatten the tensors, has no effect on the batch size
        self.flatten = tf.keras.layers.Flatten()
        
        # outputlayer
        self.out = MyDenseLayer(10, activation='softmax', kernel_regularizer=kernel_reg, bias_regularizer=bias_reg)


    @tf.function
    def call(self, inputs):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        x = self.convo(inputs)
        x = self.maxpool(x)
        x = self.convo2(x)
        x = self.flatten(x)
        return self.out(x)
