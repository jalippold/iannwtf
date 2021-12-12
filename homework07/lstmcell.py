import tensorflow as tf

class LSTM_Cell(tf.keras.layers.Layer):

    def __init__(self, units):
        super(LSTM_Cell, self).__init__()
        self.units = units
        # special initializer for the forget gates
        self.forget_dense = tf.keras.layers.Dense(units=units, activation=tf.nn.sigmoid, bias_initializer='ones')
        self.input_dense = tf.keras.layers.Dense(units=units, activation=tf.nn.sigmoid)
        self.csc_dense = tf.keras.layers.Dense(units=units, activation=tf.nn.tanh)
        self.output_dense = tf.keras.layers.Dense(units=units, activation=tf.nn.sigmoid)

    @tf.function
    def call(self, x, states):
        # states first dimension should have 2 elements
        # states[0] -> hidden state, states[1] -> cell state
        input_shape = tf.shape(x)

        hidden_state = states[0]
        cell_state = states[1]
        
        x = tf.concat([x, hidden_state], axis=1)

        forget = self.forget_dense(x)
        input = self.input_dense(x)
        cell_state_cand = self.csc_dense(x)
        output = self.output_dense(x)

        cell_state = tf.multiply(cell_state, forget)
        cell_state = tf.add(cell_state, tf.multiply(input, cell_state_cand))

        hidden_state = tf.multiply(tf.tanh(cell_state), output)

        # outputs for each state shape (batch_size, state)
        return (hidden_state, cell_state)


# Test it
"""cell = LSTM_Cell(8)
print(cell(tf.random.uniform(shape=(2,16), minval=-1, maxval=1, dtype=tf.float32),
    (tf.random.uniform(shape=(8,), minval=-1, maxval=1, dtype=tf.float32), tf.random.uniform(shape=(8,), minval=-1, maxval=1, dtype=tf.float32))))"""