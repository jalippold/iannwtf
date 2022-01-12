import tensorflow as tf


class Generator(tf.keras.Model):
    """
    This class represents Generator of the GAN
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.net_layers = []
        # initial Dense-Layer
        self.net_layers.append(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Reshape((7,7,256)))
        # use upsampling
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        # last convlayer with tanh
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))


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
