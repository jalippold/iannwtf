import tensorflow as tf


class InputEmbeddingLayer(tf.keras.layers.Layer):
    
    def __init__(self, seq_len, vocab_size, embed_dim):
        super(InputEmbeddingLayer, self).__init__()
        self.seq_len = seq_len

        self.pos_embed = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=embed_dim)

        self.token_embed = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

    @tf.function
    def call(self, input):
        pos_code_indices = tf.range(0, self.seq_len)

        # call both embedding-layers
        input_e = self.token_embed(input)
        pos_e = self.pos_embed(pos_code_indices)
        # return the sum of both 
        # return tf.keras.layers.Add()([input_e, pos_e])
        return tf.add(input_e, pos_e)