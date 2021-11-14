import tensorflow_datasets as tfds
import tensorflow as tf
from myNetworkModel import MyModel
import numpy as np
import matplotlib.pyplot as plt





def onehotify(tensor):
    """
    function taken from the exercise sheet
    :param tensor: takes a dna-string-tensor as input
    :return: returns a one-hot-encoded-version of the dna-tensor
    """
    vocab = {'A': '1', 'C': '2', 'G': '3', 'T': '0'}
    for key in vocab.keys():
        tensor = tf.strings.regex_replace(tensor, key, vocab[key])
    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot


def prepare_genomics_data(data_set):
    """
    This function prepares a dataset for our scenario
    :param data_set: the given dataset
    :return: the pipelined version of the dataset
    """
    # make one-hot-tensors of the input and the label
    data_set = data_set.map(lambda input_tensor, target_tensor: (onehotify(input_tensor), tf.one_hot(target_tensor,depth=10)))
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    data_set = data_set.cache()
    # shuffle, batch, prefetch
    data_set = data_set.shuffle(1000)
    data_set = data_set.batch(32)
    data_set = data_set.prefetch(20)
    # return preprocessed dataset
    return data_set


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
        prediction = model(input)
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
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy


if __name__ == "__main__":
    #loading the genomics dataset
    train_ds, test_ds = tfds.load('genomics_ood', split=['train', 'test'], as_supervised=True)
    # apply the data pipeline to the dataset
    prepared_train_dataset = train_ds.apply(prepare_genomics_data)
    prepared_test_dataset = test_ds.apply(prepare_genomics_data)
    # initialize the model; one input layer with 1000 inputs
    model = MyModel((1,1000))
    # clean the session
    tf.keras.backend.clear_session()
    # We only take 100.000 training examples and 1000 test_datasets
    small_prepared_train_dataset = prepared_train_dataset.take(1000)
    small_prepared_test_dataset= prepared_test_dataset.take(1000)
    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.1
    # Initialize the loss: categorical cross entropy.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters.
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []
    # testing once before we begin
    test_loss, test_accuracy = test(model, small_prepared_test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    # check how model performs on train data once before we begin
    train_loss, _ = test(model, small_prepared_train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)
    #   We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in small_prepared_test_dataset:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, small_prepared_test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
    # Visualize accuracy and loss for training and test data.
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3), ("training", "test", "test accuracy"))
    plt.show()
