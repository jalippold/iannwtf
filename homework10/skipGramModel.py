import tensorflow as tf


class SkipGramModel(tf.keras.layers.Layer):
    """
    https://www.tensorflow.org/tutorials/text/word2vec
    """
    def __init__(self, vocab, embed_sz=32):
        super(SkipGramModel, self).__init__()

        self.loss = tf.nn.nce_loss()

        self.embedding_size = embed_sz

        self.vocab = vocab


    def build(self, input_shape):
        pass

    @tf.function
    def call(self, inputs):
        pass