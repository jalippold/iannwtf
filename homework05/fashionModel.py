import tensorflow as tf

# this class represents a model which consists of
# 1 convolutional layer + one max pooling layer + one dense layer + one output layer


class MyModel(tf.keras.Model):

    def __init__(self, kernel_reg=None, bias_reg=None, dropout=False, dropout_rate= 0.2):
        super(MyModel, self).__init__()
        #first hidden layer 2DConvoLayer
        self.convo = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='same',
                                            activation='ReLU')
        # second hidden layer with pooling
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')
        # layer to flatten the tensors, has no effect on the batch size
        self.flatten = tf.keras.layers.Flatten()
        #third hidden layer; dense layer with 100 neurons
        self.dense1 = tf.keras.layers.Dense(100,activation='ReLU', use_bias=True)
        # outputlayer
        self.out = tf.keras.layers.Dense(10,activation='softmax', use_bias=True)

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
