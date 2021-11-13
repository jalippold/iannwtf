import tensorflow as tf

# custom DenseLaxer with sigmoid activation function
# based on the example from the courseware


class MyDenseLayer(tf.keras.layers.Layer):

    def __init__(self, units=8):
        super(MyDenseLayer, self).__init__()
        self.units = units
        self.activation = tf.nn.sigmoid

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x
