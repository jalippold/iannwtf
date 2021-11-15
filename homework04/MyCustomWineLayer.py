import tensorflow as tf

# custom DenseLaxer with sigmoid activation function
# based on the example from the courseware


class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units=8, kernel_regularizer=None,bias_regularizer=None):
        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = tf.nn.sigmoid
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True,
                               regularizer=self.kernel_regularizer)
        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True,
                               regularizer=self.bias_regularizer)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x
