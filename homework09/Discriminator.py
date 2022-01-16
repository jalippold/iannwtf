import tensorflow as tf


class Discriminator(tf.keras.Model):
    """
    This class represents Discriminator of the GAN
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        # self.net_layers = []
        # self.net_layers.append(tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='valid',
        #                                              input_shape=[28, 28, 1]))
        # self.net_layers.append(tf.keras.layers.BatchNormalization())
        # self.net_layers.append(tf.keras.layers.LeakyReLU())
        # self.net_layers.append(tf.keras.layers.Dropout(0.2))
        # self.net_layers.append(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        # self.net_layers.append(tf.keras.layers.BatchNormalization())
        # self.net_layers.append(tf.keras.layers.LeakyReLU())
        # self.net_layers.append(tf.keras.layers.Dropout(0.2))
        # self.net_layers.append(tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        # self.net_layers.append(tf.keras.layers.BatchNormalization())
        # self.net_layers.append(tf.keras.layers.LeakyReLU())
        # self.net_layers.append(tf.keras.layers.Dropout(0.2))
        # self.net_layers.append(tf.keras.layers.Flatten())
        # # Dense classification layer
        # self.net_layers.append(tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid))

        self.net_layers = []
        self.net_layers.append(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                                     input_shape=[28, 28, 1]))
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Dropout(0.3))
        self.net_layers.append(
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.net_layers.append(tf.keras.layers.LeakyReLU())
        self.net_layers.append(tf.keras.layers.Dropout(0.3))
        self.net_layers.append(tf.keras.layers.Flatten())
        # Dense classification layer
        self.net_layers.append(tf.keras.layers.Dense(1))

        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        # tf.keras.metrics.BinaryAccuracy, 
                        ]

    # should return values near1 for real images and 0 for fake images
    @tf.function
    def call(self, inputs, training=False):
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
    def train_step(self, real_images, generated_images):
        with tf.GradientTape() as tape:
            real_output = self(real_images, training=True)
            fake_output = self(generated_images, training=True)
            real_targets = tf.ones_like(real_output)
            fake_targets = tf.zeros_like(fake_output)
            
            real_loss = self.loss(real_targets, real_output)
            fake_loss = self.loss(fake_targets, fake_output)
            total_loss = real_loss + fake_loss
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics_list[0].update_state(total_loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics_list[1:]:
            metric.update_state(y_true=real_targets, y_pred=real_output)
            metric.update_state(y_true=fake_targets, y_pred=fake_output)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics_list}

    @tf.function
    def test_step(self, real_images, generated_images):

        real_output = self(real_images, training=False)
        fake_output = self(generated_images, training=False)
        real_targets = tf.ones_like(real_output)
        fake_targets = tf.zeros_like(fake_output)
        
        real_loss = self.loss(real_targets, real_output)
        fake_loss = self.loss(fake_targets, fake_output)
        total_loss = real_loss + fake_loss
                
        self.metrics_list[0].update_state(total_loss)
        
        for metric in self.metrics_list[1:]:
            metric.update_state(y_true=real_targets, y_pred=real_output)
            metric.update_state(y_true=fake_targets, y_pred=fake_output)

        return {m.name: m.result() for m in self.metrics_list}