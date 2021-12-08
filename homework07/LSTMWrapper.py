import tensorflow as tf
from LSTMCell import CustomLSTMCell

class LSTMWrapper(tf.keras.layers.Layer):
    def __init__(self, LSTMCell, return_sequences=False):
        super(LSTMWrapper, self).__init__()
        self.return_sequences = return_sequences
        self.cell = LSTMCell

    def call(self, data):
        length = data.shape[1]
        # initialize state of the lstm cell
        state = tf.zeros((data.shape[0], self.cell.units), tf.float32)
        # initialize cell-state of the lstm cell
        cell_state = tf.zeros((data.shape[0], self.cell.units), tf.float32)
        # initialize array for hidden states (only relevant if self.return_sequences == True)
        hidden_states = tf.TensorArray(dtype=tf.float32, size=length)
        # initialize array for cell states (only relevant if self.return_sequences == True)
        cell_states = tf.TensorArray(dtype=tf.float32, size=length)
        for t in tf.range(length):
            input_t = data[:, t, :]
            state, cell_state = self.cell(input_t, state, cell_state)
            if self.return_sequences:
                # write the states to the TensorArray
                hidden_states = hidden_states.write(t, state)
                cell_states = cell_states.write(t, state)
        if self.return_sequences:
            # transpose the sequence of hidden_states from TensorArray accordingly
            # (batch and time dimensions are otherwise switched after .stack())
            outputs = tf.transpose(hidden_states.stack(), [1, 0, 2])
        else:
            # take the last hidden state of the simple rnn cell
            outputs = state
        return outputs