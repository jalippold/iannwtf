import tensorflow as tf
from resBlock import MyResBlock


class MyResModel(tf.keras.Model):
    """
    This class represents a model
    """

    def __init__(self):
        super(MyResModel, self).__init__()

        self.net_layers = []
        # initial Layer
        self.net_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
        # resLayers
        self.net_layers.append(MyResBlock(number_filters=64, number_out_filters=256, mode="normal"))
        self.net_layers.append(MyResBlock(number_filters=128, number_out_filters=256, mode="strided"))
        self.net_layers.append(MyResBlock(number_filters=256, number_out_filters=256, mode="constant"))
        # global average pooling layer
        self.net_layers.append(tf.keras.layers.GlobalAveragePooling2D())
        # outputlayer
        self.net_layers.append(tf.keras.layers.Dense(10, activation='softmax'))

    @tf.function
    def call(self, inputs):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        for layer in self.net_layers:
            inputs = layer(inputs)
        return inputs