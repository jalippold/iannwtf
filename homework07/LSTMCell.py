import tensorflow as tf


class CustomLSTMCell(tf.keras.layers.Layer):
    def __init__(self, units, kernel_regularizer=None):
        super(CustomLSTMCell, self).__init__()
        self.units = units
        self.dense_hstate = tf.keras.layers.Dense(units,kernel_regularizer=kernel_regularizer,activation='sigmoid')
        self.dense_input = tf.keras.layers.Dense(units, kernel_regularizer=kernel_regularizer,activation='sigmoid')
        self.dense_output = tf.keras.layers.Dense(units, kernel_regularizer=kernel_regularizer,activation='sigmoid')
        # setting bias of forget gate to one
        self.dense_forget = tf.keras.layers.Dense(units, kernel_regularizer=kernel_regularizer, bias_initializer='ones',activation='sigmoid')
        self.dense_cell_state = tf.keras.layers.Dense(units, kernel_regularizer=kernel_regularizer, activation='tanh')
        #self.bias_f = tf.Variable(tf.ones(units), name="LSTM_Cell_biases_f")
        #self.bias_i = tf.Variable(tf.zeros(units), name="LSTM_Cell_biases_i")
        #self.bias_c = tf.Variable(tf.zeros(units), name="LSTM_Cell_biases_c")
        #self.bias_o = tf.Variable(tf.zeros(units), name="LSTM_Cell_biases_o")
        self.state_size = units

    def call(self, input_t, state,cell_state):
        f_t = self.dense_forget(tf.concat([state, input_t], axis=0))
        i_t = self.dense_input(tf.concat([state, input_t], axis=0))
        cell_cand = self.dense_cell_state(tf.concat([state, input_t], axis=0))
        c_t = tf.math.add(tf.math.multiply(f_t, cell_state), tf.math.multiply(i_t, cell_cand))
        o_t = self.dense_output(tf.concat([state, input_t], axis=0))
        h_t = tf.math.multiply(o_t, tf.nn.tanh(c_t))
        #return new hidden state as output
        return h_t, c_t
