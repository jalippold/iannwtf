import tensorflow as tf

# custom DenseLaxer with sigmoid activation function
# based on the example from the courseware


class MyResBlock(tf.keras.layers.Layer):

    def __init__(self, number_filters, number_out_filters, mode):
        super(MyResBlock, self).__init__()
        self.number_filters = number_filters
        self.number_out_filters = number_out_filters
        self.mode = mode
        # creating the needed layers
        self.BatchNormal = tf.keras.layers.BatchNormalization()
        self.Conv2Din = tf.keras.layers.Conv2D(filters=number_filters, kernel_size =(1, 1))
        self.Conv2Dout = tf.keras.layers.Conv2D(filters=number_out_filters, kernel_size=(1, 1))
        self.Conv2Dnormal = tf.keras.layers.Conv2D(filters=number_filters, kernel_size =(3, 3), padding="same")
        self.Conv2Dstride = tf.keras.layers.Conv2D(filters=number_filters, kernel_size =(3, 3), padding="same", strides=(2,2))
        self.Conv2Dstrideconstant = tf.keras.layers.Conv2D(filters=number_filters, kernel_size =(3, 3), padding="same")
        self.MaxPool = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=(2, 2))
        self.Relulayer = tf.keras.layers.Activation(tf.nn.relu)



    @tf.function
    def call(self, inputs):
        # use batch normalization and a non-linearity (relu)
        x_out = self.BatchNormal(inputs)
        x_out = self.Relulayer(x_out)
        x_out = self.Conv2Din(x_out)
        x_out = self.BatchNormal(x_out)
        x_out = self.Relulayer(x_out)
        if self.mode == "normal":
            x_out = self.Conv2Dnormal(x_out)
            # transform original input to also have 256 channels
            x = self.Conv2Dout(inputs)
        elif self.mode == "strided":
            # set number of output channels to match number of input channels (else we need a 1x1 convolution)
            #TODO out_filters = inputs.shape[-1]
            # do strided convolution (reducing feature map size)
            x_out = self.Conv2Dstride(x_out)
            # transform original input with 1x1 strided max pooling to match output shape
            x = self.MaxPool(inputs)
        elif self.mode == "constant":
            x_out = self.Conv2Dstrideconstant(x_out)

        # calculations for all cases
        x_out = self.BatchNormal(x_out)
        x_out = self.Relulayer(x_out)
        x_out = self.Conv2Dout(x_out)
        # Add x and x_out
        x_out = tf.keras.layers.Add()([x_out, x])
        return x_out