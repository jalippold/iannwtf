from TransformerBlock import TransformerBlock
from InputEmbeddingLayer import InputEmbeddingLayer

import tensorflow as tf


class MyNLPModel(tf.keras.Model):

    def __init__(self, tokenizer, seq_len, vocab_size, embed_dim, dense_dim):
        super(MyNLPModel, self).__init__()

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                        ]

        self.input_embed = InputEmbeddingLayer(seq_len, vocab_size, embed_dim)
        self.trans_block = TransformerBlock(embed_dim, dense_dim)
        self.glob_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dense = tf.keras.layers.Dense(units=vocab_size, activation=None)
    
    @tf.function
    def call(self, input):
        x = self.input_embed(input)
        x = self.trans_block(x)
        x = self.glob_pool(x)
        x = self.dense(x)
        return x


    def reset_metrics(self):    
        for metric in self.metrics_list:
            metric.reset_states()


    @tf.function
    def train_step(self, data):
        x, targets = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(targets,predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    def gen_text(self, prompt, top_k):
        tokens = self.tokenizer.tokenize(prompt)
        input_len = len(tokens)
        while self.seq_len >= input_len:
            padding = [-1 for _ in range(self.seq_len-input_len)]
            padded_input = tf.expand_dims(tf.concat(tf.convert_to_tensor(padding, dtype=tf.int32), tokens), 0)
            scores = self(padded_input, training=False)

            values, indices = tf.math.top_k(scores, top_k)
            sample_ind = tf.random.categorical(values, 1)
            tokens = tf.concat(tokens, indices[sample_ind])

            input_len += 1

        return self.tokenizer.detokenize(tokens)