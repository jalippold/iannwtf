import tensorflow as tf
from tensorflow.python.keras.engine import training


class DenseBlock(tf.keras.layers.Layer):
    """Class encapsulating a Dense Block"""

    def __init__(self, new_channels=32, len_dense=2, intern_filters=[32], kernel_sizes=[(3,3), (3,3)]):
        """
        new_channels:   Number of channels added to the input by this Dense Block
        len_dense:      Number of convolution layers in the dense block
        intern_filters  Number of filters for each of the first len_dense-1 layers (last one has #new_channels filter)
        kernel_sizes    Dimensions of the kernels of each convolution layer
        """
        super(DenseBlock, self).__init__()

        if len(intern_filters)+1 != len_dense or len(kernel_sizes) != len_dense:
            print(intern_filters, kernel_sizes)
            raise ValueError(f"Length of filters or kernels does not match depth of dense layer! {len_dense},{len(intern_filters)},{len(kernel_sizes)}")

        self.batchnorms = []
        self.activations = []
        self.dconvs = []
        intern_filters.append(new_channels)
        self.dense_depth = len_dense

        for i in range(self.dense_depth):
            self.batchnorms.append(tf.keras.layers.BatchNormalization())
            self.activations.append(tf.keras.layers.Activation(tf.nn.relu))
            self.dconvs.append(tf.keras.layers.Conv2D(filters=intern_filters[i], kernel_size=kernel_sizes[i], padding="same"))

        self.concat = tf.keras.layers.Concatenate(axis=-1)


    @tf.function
    def call(self, inputs):
        dout = inputs

        for i in range(self.dense_depth):
            dout = self.batchnorms[i](dout)
            dout = self.activations[i](dout)
            dout = self.dconvs[i](dout)

        return self.concat([inputs, dout])