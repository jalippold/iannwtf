import tensorflow as tf


class Generator(tf.keras.Model):
    """
    This class represents Generator of the GAN
    """

    def __init__(self, input_dim=100):
        super(Generator, self).__init__()

        self.net_layers = []
        # initial Dense-Layer
        self.net_layers.append(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(input_dim,)))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Reshape((7,7,256)))
        # use upsampling
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(128, (6, 6), strides=(2, 2), padding='same', use_bias=False)) # hints state that Conv2DTrans might work better with even kernel-size
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding='same', use_bias=False)) # hints state that Conv2DTrans might work better with even kernel-size
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        # last convlayer with tanh
        self.net_layers.append(tf.keras.layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))


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
            inputs = layer(inputs, training=training)
        return inputs
