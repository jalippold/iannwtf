import tensorflow as tf
from resBlock import MyResBlock


class MyResModel(tf.keras.Model):
    """
    This class represents a model which consists of a convolutional layer + one max pooling layer + one flattening layer
    + one dense layer + one output layer
    """

    def __init__(self):
        super(MyResModel, self).__init__()
        # initial Layer
        self.Conv2D_initial = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")
        # resLayers
        self.res_normal = MyResBlock(number_filters=64, number_out_filters=256, mode="normal")
        self.res_strided= MyResBlock(number_filters=128, number_out_filters=256, mode="strided")
        self.res_constant = MyResBlock(number_filters=256, number_out_filters=256, mode="constant")
        # outputlayer
        self.out = tf.keras.layers.Dense(10, activation='softmax')

    @tf.function
    def call(self, inputs):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        x = self.Conv2D_initial(inputs)
        x = self.res_normal(x)
        x = self.res_strided(x)
        x = self.res_constant(x)
        x = self.out(x)
        return x