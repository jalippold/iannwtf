import tensorflow as tf
from denseBlock import DenseBlock
from transitionLayer import TransitionLayer

class DenseNet(tf.keras.Model):
    """Class Representing a DenseNet model with two DenseBlocks"""
    def __init__(self):
        super(DenseNet, self).__init__()

        self.conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding="same")

        self.denseblock1 = DenseBlock(new_channels=32, len_dense=2, intern_filters=[32], kernel_sizes=[(3,3),(3,3)])
        self.trans1 = TransitionLayer(num_filters=64, pooling_strides=(2,2))
        self.denseblock2 = DenseBlock(new_channels=64, len_dense=2, intern_filters=[64], kernel_sizes=[(2,2),(2,2)])
        self.trans2 = TransitionLayer(num_filters=32, pooling_strides=(4,4))

        self.flatten = tf.keras.layers.Flatten()

        self.dense = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)


    def call(self, x):
        x = self.conv(x)
        x = self.denseblock1(x)
        x = self.trans1(x)
        x = self.denseblock2(x)
        x = self.trans2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.out(x)
