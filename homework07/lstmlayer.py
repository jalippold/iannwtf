import tensorflow as tf
from lstmcell import LSTM_Cell


class LSTM_Layer(tf.keras.layers.Layer):
    def __init__(self, cell):
        super(LSTM_Layer, self).__init__()
        # start with single cell layer
        self.cell = cell

    # input here over multiple time steps
    # and returns output over multiple time steps
    def call(self, x, states):
        # input should have (batch_size, seq_len, input_size]
        input_shape = tf.shape(x)
        states = [states]
        # creates a list which contains batch_size times lists
        # with the length of seq_len
        outputs = [[None for t in range(input_shape[1])] for batch in range(input_shape[0])]

        # for t in range(seq_len)
        for t in range(input_shape[1]):
            # x[:,t,:] --> input at time t + input last state [hidden state, cell state]
            ret = self.cell(x[:,t,:], states[-1])
            states.append(ret)
            # for dim in range batch_size
            for dim in range(input_shape[0]):
                outputs[dim][t] = states[-1][0]
        
        return outputs

    
    def zero_states(self, batch_size):
        return (tf.zeros(shape=(batch_size, self.cell.units)), tf.zeros(shape=(batch_size, self.cell.units)))


BATCH_SIZE = 2
SEQ_LEN = 3
INPUT_UNITS = 16

cell = LSTM_Cell(8)
layer = LSTM_Layer(cell)

#x = tf.random.uniform(shape=(BATCH_SIZE, SEQ_LEN, INPUT_UNITS), minval=-1, maxval=1)
#print(x)
#states = layer.zero_states(BATCH_SIZE)

# we get an output of shape (batch_size, seq_len, <Tensor of size units>). is that what we want?!
#print(layer(x, states))