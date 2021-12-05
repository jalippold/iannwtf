import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from denseBlock import DenseBlock
from transitionLayer import TransitionLayer

class DenseNet(tf.keras.Model):
    """Class Representing a DenseNet model with two DenseBlocks"""
    def __init__(self):
        super(DenseNet, self).__init__()

        self.net_layers = []

        self.net_layers.append(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same"))

        self.net_layers.append(DenseBlock(new_channels=32, len_dense=2, intern_filters=[32], kernel_sizes=[(3,3),(3,3)]))
        self.net_layers.append(TransitionLayer(num_filters=(32+32)//2, pooling_strides=(2,2)))
        self.net_layers.append(DenseBlock(new_channels=64, len_dense=2, intern_filters=[64], kernel_sizes=[(3,3),(3,3)]))
        self.net_layers.append(TransitionLayer(num_filters=(32+64)//2, pooling_strides=(4,4)))
        self.net_layers.append(DenseBlock(new_channels=64, len_dense=2, intern_filters=[64], kernel_sizes=[(3,3),(3,3)]))

        # hints mentioned after the DenseBlocks there should follow a BatchNormalization-layer and a GlobalAveragePooling-layer
        self.net_layers.append(BatchNormalization())
        self.net_layers.append(tf.keras.layers.Flatten())
        # self.net_layers.append(GlobalAveragePooling2D())

        self.net_layers.append(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
        self.net_layers.append(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    @tf.function
    def call(self, x):
        for layer in self.net_layers:
            x = layer(x)
        return x
