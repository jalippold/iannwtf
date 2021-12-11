import tensorflow as tf
from tensorflow.python.framework.tensor_spec import TensorSpec
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse
from MyModel import LSTMModel
SEQ_LEN = 25
NUM_SAMPLES = 400

MIN_VAL = -1
MAX_VAL = 1

def integration_task(seq_len, num_samples):
    for i in range(num_samples):
        input = tf.random.uniform(shape=(seq_len, 1), minval=MIN_VAL, maxval=MAX_VAL, dtype=tf.float32)
        target = 1 if tf.math.reduce_sum(input, axis=0) > 0 else 0
        yield (input, tf.constant(target, dtype=tf.float32,shape=(1)))


def my_integration_task():
    for data in integration_task(SEQ_LEN, NUM_SAMPLES):
        yield data

# (seq_len)(3)

#ds = tf.data.Dataset.from_generator(my_integration_task, output_signature=(tf.TensorSpec(shape=(SEQ_LEN,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)))


################################
# Building Dataset done!
################################



def prepare_myds(myds):
    # we don't have to do that much, we build the dataset like we want it...

    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    myds = myds.cache()
    # shuffle, batch, prefetch
    myds = myds.shuffle(1000)
    myds = myds.batch(64,drop_remainder=True)
    myds = myds.prefetch(32)
    # return preprocessed dataset
    return myds


#@tf.function
def train_step(model, input, target, loss_function, optimizer):
    """
    This function executes a training step on the given network.
    Using gradient descend
    :param model: the network model
    :param input: the given input tensors
    :param target: the given target tensors
    :param loss_function: the given loss function
    :param optimizer: the given optimizer
    :return: the loss during this trainin step
    """
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input, training=True)
        loss = loss_function(target, prediction)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_data, loss_function):
    """
    # Tests over complete test data
    :param model: the network model
    :param test_data: the given input tensors
    :param loss_function: the given loss function
    :return: the test loss and test accuracy
    """
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (input, target) in test_data:
        #print(input)
        prediction = model(input, training=False)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = target == np.round(prediction)
        #print(sample_test_accuracy)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_accuracy_aggregator.append(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy



if __name__ == "__main__":
    #parse arguments to determine which model should be used
    #parser = argparse.ArgumentParser(description='ResNet or DenseNet')
    #parser.add_argument('--model', type=str, help='ResNet of DenseNet', required=True)
    #args = parser.parse_args()

    tf.keras.backend.clear_session()
    
    # loading the data set
    train_ds = tf.data.Dataset.from_generator(my_integration_task, 
            output_signature=(tf.TensorSpec(shape=(SEQ_LEN,1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
    test_ds = tf.data.Dataset.from_generator(my_integration_task, 
            output_signature=(tf.TensorSpec(shape=(SEQ_LEN,1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32)))
    train_ds = train_ds.apply(prepare_myds)
    test_ds = test_ds.apply(prepare_myds)
    #print(test_ds.take(10))
    #print(train_ds.take(10))
    #train_ds = train_ds.apply(prepare_myds)
    #train_ds.batch(64)
    #test_ds = test_ds.apply(prepare_myds)
    #test_ds.batch(64)
    #print(train_ds.take(1))
    # test preprocessing
    #for elem in train_ds:
    #    print(elem)
    #    break
    ### Hyperparameters
    num_epochs = 30
    learning_rate = tf.constant(0.001, dtype=tf.float32)
    # Initialize the loss: categorical cross entropy.
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
    # Initialize the optimizer: SGD with default parameters.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    #create the model: TODO!
    hidden_state_size = 128
    model = LSTMModel(units=hidden_state_size, batchsize = 64)
    # testing once before we begin
    test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    # check how model performs on train data once before we begin
    train_loss, _ = test(model, train_ds, cross_entropy_loss)
    train_losses.append(train_loss)
    #   We train for num_epochs epochs.
    for epoch in range(num_epochs):
        start_time = datetime.datetime.now()
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in train_ds:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_ds, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        diff_time = datetime.datetime.now() - start_time
        print(f"Epoch {epoch} took {diff_time} to complete.")

    print(model.summary())
    # Visualize accuracy and loss for training and test data.
    plt.subplot(1, 1, 1)
    line1, = plt.plot(train_losses, '-x')
    line2, = plt.plot(test_losses, '-+')
    line3, = plt.plot(test_accuracies, '-o')
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3), ("training loss", "test loss", "test accuracy"))
    plt.show()