import tensorflow as tf


class EncoderModel(tf.keras.Model):
    """
    This class represents the Encoder Model
    """

    def __init__(self):
        super(EncoderModel, self).__init__()

        self.net_layers = []
        # initial Layer
        # reduce feature map size
        self.net_layers.append(tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=2 , padding="same", activation="relu"))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        # second Conv2D Layer
        self.net_layers.append(tf.keras.layers.Conv2D(filters=8, kernel_size=3, strides=2, padding="same", activation="relu"))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        # flatten
        self.net_layers.append(tf.keras.layers.Flatten())
        # Embedding layer
        self.net_layers.append(tf.keras.layers.Dense(10, activation='sigmoid'))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

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