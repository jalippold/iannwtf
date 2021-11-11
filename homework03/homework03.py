import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import one_hot
from tensorflow.python.ops.gen_math_ops import tanh_grad
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense

import numpy as np

########################
# Task 1: Dataset
########################

# load not working, some issues with anacondas ssl certificates
test_ds, train_ds = tfds.load('genomics_ood', split=('test[:1000]', 'train[:100000]'), try_gcs=False)


base_to_val = {
    b'A': 0,
    b'C': 1,
    b'G': 2,
    b'T': 3
}

def str_to_onehot(elem):
    # TODO SOMEHOW!!!!! get the string value of tensor elem['seq'] and append a one_hot Tensor for each base to list one_hot
    one_hot = []
    for base in elem['seq'].numpy(): # does not work, how does one iterate over the string encapsulated in tensor elem['seq']?!!!!!
        one_hot.append(tf.one_hot(base_to_val[base], depth=4, dtype=tf.uint8))
    elem['seq'] = tf.concat(one_hot, axis=0)
    elem['label'] = tf.one_hot(elem['label'], depth=10, dtype=tf.uint8)
    return elem
    

def prepare_genomics_data(data):
    # transform string encoded input and output into one-hot encoding
    data = data.map(str_to_onehot)
    # shuffle, batch and prefetch data
    data = data.cache()
    data = data.shuffle(1000).batch(8).prefetch(20)
    return data

train_ds.apply(prepare_genomics_data)
test_ds.apply(prepare_genomics_data)




########################
# Task 2: Model
########################

class MyDense(tf.keras.layers.Layer):
    def __init__(self, units=8, activation=tf.nn.sigmoid):
        super(MyDense, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='random_normal',
                                    trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                    initializer='random_normal',
                                    trainable=True)

    def call(self, inputs):
        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)
        return x


class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = MyDense(256, activation=tf.nn.sigmoid)
        self.dense2 = MyDense(256, activation=tf.nn.sigmoid)
        self.out = MyDense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x



########################
# Task 3: Training
########################


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for elem in test_data:
        prediction = model(elem['seq'])
        sample_test_loss = loss_function(elem['label'], prediction)
        sample_test_accuracy =  np.argmax(elem['label'], axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def train_step(model, input, target, loss_function, optimizer):
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


learning_rate = 0.1
num_epochs = 10

model = MyModel()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

#testing once before we begin
test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = test(model, train_ds, cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg = []
    for elem in train_ds:
        train_loss = train_step(model, elem['seq'], elem['label'], cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

########################
# Task 4: Visualization
########################