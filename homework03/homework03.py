import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense

import numpy as np

########################
# Task 1: Dataset
########################


# load not working, some issues with anacondas ssl certificates
train_ds, test_ds = tfds.load('genomics_ood', split=['train', 'test'])
train_ds = train_ds.take(100000)
test_ds = test_ds.take(1000)

str_to_onehot = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

def prepare_genomics_data(data):
    # transform string encoded input and output into one-hot encoding
    data = data.map(lambda inpt, target: (tf.one_hot(str_to_onehot[inpt], depth=4), tf.one_hot(target, depth=10)))
    # shuffle, batch and prefetch data
    data = data.shuffle(1000).batch(8).prefetch(20)
    return data

train_ds = prepare_genomics_data(train_ds)
test_ds = prepare_genomics_data(test_ds)

########################
# Task 2: Model
########################


# TODO create own Dense Layer class...


class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = Dense(256, activation=tf.nn.sigmoid)
        self.dense2 = Dense(256, activation=tf.nn.sigmoid)
        self.out = Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x




########################
# Task 3: Model
########################


def test(model, test_data, loss_function):
    # test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
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


