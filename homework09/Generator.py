import tensorflow as tf


class Generator(tf.keras.Model):
    """
    This class represents Generator of the GAN
    """

    def __init__(self, input_dim=100):
        super(Generator, self).__init__()

        self.net_layers = []
        # initial Dense-Layer
        self.net_layers.append(tf.keras.layers.Dense(7*7*256, use_bias=True, input_shape=(input_dim,)))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Reshape((7,7,256)))
        # use upsampling
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(128, (6, 6), strides=(2, 2), padding='same', use_bias=True)) # hints state that Conv2DTrans might work better with even kernel-size
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Conv2DTranspose(64, (6, 6), strides=(2, 2), padding='same', use_bias=True)) # hints state that Conv2DTrans might work better with even kernel-size
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        # last convlayer with tanh
        self.net_layers.append(tf.keras.layers.Conv2D(1, (5, 5), strides=(1, 1), padding='same', use_bias=True, activation='tanh'))

        self.optimizer = tf.keras.optimizers.Adam(1e-2)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        # tf.keras.metrics.BinaryAccuracy, 
                        ]
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

    def reset_metrics(self):    
        for metric in self.metrics_list:
            metric.reset_states()

    @tf.function
    def train_step(self, noise, discriminator):
        with tf.GradientTape() as tape:
            generated_images = self(noise, training=True)
            fake_output = discriminator(generated_images, training=False)
            fake_targets = tf.ones_like(fake_output)

            total_loss = self.loss(fake_targets, fake_output)
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics_list[0].update_state(total_loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics_list[1:]:
            metric.update_state(fake_targets, fake_output)

        # Return a dictionary mapping metric names to current value
        return ({m.name: m.result() for m in self.metrics_list}, generated_images)

    # @tf.function
    # def test_step(self, data):

    #     x, targets = data
        
    #     predictions = self(x, training=False)
        
    #     loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
    #     self.metrics_list[0].update_state(loss)
        
    #     for metric in self.metrics_list[1:]:
    #         metric.update_state(targets, predictions)

    #     return {m.name: m.result() for m in self.metrics_list}