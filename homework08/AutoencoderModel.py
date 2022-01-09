import tensorflow as tf
from Encoder import EncoderModel
from Decoder import DecoderModel


class AutoEncoderModel(tf.keras.Model):
    """
    This class represents the Autoencoder Model
    """

    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        self.encoder = EncoderModel()
        self.decoder = DecoderModel()

    @tf.function
    def call(self, inputs):
        """
        calculates the output of the network for
        the given input
        :param inputs: the input tensor of the network
        :return: output of the network
        """
        x = self.encoder(inputs)
        x = self.decoder(x)
        return x