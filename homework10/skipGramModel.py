from numpy import dtype
import tensorflow as tf


class SkipGramModel(tf.keras.Model):
    """
    https://www.tensorflow.org/tutorials/text/word2vec
    """
    def __init__(self, vocab_sz, embed_sz=32):
        super(SkipGramModel, self).__init__()
        self.embedding_target = tf.keras.layers.Embedding(vocab_sz, embed_sz, input_length=1)
        self.embedding_context = tf.keras.layers.Embedding(vocab_sz, embed_sz, input_length=5) # num_negatives + 1
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.vocab_size = vocab_sz

        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        # tf.keras.metrics.BinaryAccuracy, 
                        ]


    """

    def build(self, input_shape):
        pass"""

    @tf.function
    def call(self, inputs, training=False):
        target, context, label = inputs
        target_emb = self.embedding_target(target, training=training)
        context_emb = self.embedding_context(context, training=training)
        #context_emb = self.embedding_target(context, training=training)
        print(target_emb)
        print(context_emb)

        # target has shape (BATCH_SIZE, EMBEDDING_SIZE), context (BATCH_SIZE, CONTEXT_SIZE, EMBEDDING_SIZE)
        # get dot product in shape (BATCH_SIZE, CONTEXT_SIZE)
        dots = tf.einsum("be, bce->bc", target_emb, context_emb)
        print(dots)
        print(label)
        return tf.math.multiply(dots, tf.cast(label, tf.float32)) # return loss for positive label, 0 for negative label -> TODO!

    @tf.function
    def train_step(self, target, context, label):
        """
        All arguments are scalars

        target: target word
        context: context word
        label: 1 -> positive sample, 0 -> negative sample
        """
        with tf.GradientTape() as tape:
            loss = self((target, context, label), training=True)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics_list[0].update_state(loss)
        
        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics_list}


    @tf.function
    def test_step(self, target, context, labels):
        loss = self((target, context, labels), training=False)

        self.metrics_list[0].update_state(loss)

        return {m.name: m.result() for m in self.metrics_list}


    def reset_metrics(self):    
        for metric in self.metrics_list:
            metric.reset_states()

    #@tf.function
    def calculate_nearest_neighbors(self, token, k):
        token_emb = self.embedding_target(tf.constant(token))
        knn = [(None, tf.constant(-10.)) for i in range(k)]

        for i in range(self.vocab_size):
            if i == token:
                continue

            emb = self.embedding_target(tf.constant(i))
            cs = tf.tensordot(token_emb, emb, axes=1)

            # first list element is smallest
            if tf.math.greater(cs, knn[0][1]):
                knn[0] = (i, cs)


            knn.sort(key=lambda x: x[1])

        # should now contain tuples of (token, cosine_similarity)
        return knn
