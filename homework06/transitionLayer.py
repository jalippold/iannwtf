import tensorflow as tf
from tensorflow.python.keras.engine import training

########################################################

########################################################

class TransitionLayer(tf.keras.layers.Layer):
    """
    Class Implementing a Transition Layer
    Used inbetween Dense blocks
    
    'num_filters' defines the channel depth of the output
    'pooling_strides' defines the output size of the data    
    """
    def __init__(self, num_filters, pooling_strides):
        super(TransitionLayer, self).__init__()

        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
        self.conv = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1,1), use_bias=False)
        self.pooling = tf.keras.layers.AveragePooling2D(strides=pooling_strides, pool_size=(2,2))

    
    def call(self, inputs):
        inputs = self.batchnorm(inputs)
        inputs = self.activation(inputs)
        inputs = self.conv(inputs)
        return self.pooling(inputs)