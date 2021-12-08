import tensorflow as ts
from lstmcell import LSTM_Cell


class LSTMWrapper(tf.keras.layers.Layer):
    def __init__(self, LSTM_Cell, return_sequences=False):
        super(LSTMWrapper, self).__init__()

        self.return_sequences = return_sequences

        self.cell = LSTM_Cell

    def call(self, data, training=False):

        length = data.shape[1]

        # initialize state of the simple rnn cell
        state = tf.zeros((data.shape[0], self.cell.units), tf.float32)

        # initialize array for hidden states (only relevant if self.return_sequences == True)
        hidden_states = tf.TensorArray(dtype=tf.float32, size=length)

        for t in tf.range(length):
            input_t = data[:, t, :]

            state = self.cell(input_t, state, training)

            if self.return_sequences:
                # write the states to the TensorArray
                # hidden_states = hidden_states.write(t, state)
                hidden_states.append(state)

        if self.return_sequences:
            # transpose the sequence of hidden_states from TensorArray accordingly
            # (batch and time dimensions are otherwise switched after .stack())
            outputs = tf.transpose(hidden_states.stack(), [1, 0, 2])

        else:
            # take the last hidden state of the simple rnn cell
            outputs = state

        return outputs