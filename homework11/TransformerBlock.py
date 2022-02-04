import tensorflow as tf

class TransformerBlock(tf.keras.layers.Layer):

    def __init__(self, embed_dim, dense_dim):
        super(TransformerBlock, self).__init__()

        self.attention_heads = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim)

        self.dense_1 = tf.keras.layers.Dense(units=dense_dim, activation="relu")
        self.dense_2 = tf.keras.layers.Dense(units=embed_dim, activation=None)

        self.dropout_1 = tf.keras.layers.Dropout(rate=0.1)
        self.dropout_2 = tf.keras.layers.Dropout(rate=0.1)

        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    @tf.function
    def call(self, input):
        x = self.attention_heads(input, input)
        x = self.dropout_1(x)
        x = tf.keras.layers.Add()([x, input])
        ln_out = self.layer_norm_1(x)

        x = self.dense_1(ln_out)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = tf.keras.layers.Add()([x, ln_out])
        x = self.layer_norm_2(x)

        return x