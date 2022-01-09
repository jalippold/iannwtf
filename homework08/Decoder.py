import tensorflow as tf


class DecoderModel(tf.keras.Model):
    """
    This class represents the Encoder Model
    """

    def __init__(self):
        super(DecoderModel, self).__init__()
        self.net_layers = []
        # initial Layer
        # reduce feature map size
        self.net_layers.append(tf.keras.layers.Dense(49, activation='sigmoid'))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        # Reshape Layer
        self.net_layers.append(tf.keras.layers.Reshape((7,7,1)))
        # first Conv2DTransposeLayer
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=3, strides=2 , padding="same", activation="relu"))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        # second Conv2DTransposeLayer
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding="same", activation="relu"))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        # Output layer
        self.net_layers.append(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding="same", activation="sigmoid"))

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