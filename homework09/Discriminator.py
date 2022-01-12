import tensorflow as tf


class Discriminator(tf.keras.Model):
    """
    This class represents Discriminator of the GAN
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net_layers = []
        self.net_layers.append(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                     input_shape=[28, 28, 1]))
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Dropout(0.3))
        self.net_layers.append(
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Dropout(0.3))
        self.net_layers.append(tf.keras.layers.Flatten())
        # Dense classification layer
        self.net_layers.append(tf.keras.layers.Dense(1))

    # should return positive values for real images and negative values for
    # fake images
    @tf.function
    def call(self, inputs, training):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :training: boolean for training
        :return: output of the network
        """
        for layer in self.net_layers:
            inputs = layer(inputs)
        return inputs
