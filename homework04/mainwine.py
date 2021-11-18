import pandas as pd
import tensorflow as tf
from myWineNetworkModel import MyModel
import numpy as np
import matplotlib.pyplot as plt


def make_binary(target):
    """
    Function which converts a target into a binary reprensentation
    :param target: input value
    :return: 1.0 if wine quality above 6, 0.0 else
    """
    if target >= 6:
        return tf.constant(1, dtype='float64')
    else:
        return tf.constant(0, dtype='float64')


def prepare_wine_data(data_set):
    """
    This function prepares a dataset for our scenario
    :param data_set: the given dataset
    :return: the pipelined version of the dataset
    """
    # make one-hot-tensors of the input and the label
    data_set = data_set.map(lambda input_tensor, target_tensor: (input_tensor, make_binary(target_tensor)))
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
        prediction = model(input, training=False)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = target == np.round(prediction,0)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy



if __name__ == "__main__":

    random_seed = 42
    # read the csv data from file
    wine_data = pd.read_csv('winequality-red.csv', sep=';', header=0)
    print('------')
    print(wine_data.loc[wine_data['quality'] >= 6])
    print('------')
    # split the dataset in train, test, validation
    wine_data_training = wine_data.sample(frac=0.7, random_state=random_seed)
    wine_data_test = wine_data.drop(wine_data_training.index)
    wine_data_validation = wine_data_test.sample(frac=0.5, random_state=random_seed)
    wine_data_test = wine_data_test.drop(wine_data_validation.index)
    # separate labels from input
    # for test data
    wine_data_test_input = wine_data_test.drop(['quality'], axis=1)
    wine_data_test_label = wine_data_test.drop(["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                                                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                                                "density", "pH", "sulphates", "alcohol"], axis=1)
    # for the training data
    wine_data_training_input = wine_data_training.drop(['quality'], axis=1)
    wine_data_training_label = wine_data_training.drop(
        ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
         "chlorides", "free sulfur dioxide", "total sulfur dioxide",
         "density", "pH", "sulphates", "alcohol"], axis=1)
    # for validation
    wine_data_validation_input = wine_data_validation.drop(['quality'], axis=1)
    wine_data_validation_label = wine_data_validation.drop(
        ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
         "chlorides", "free sulfur dioxide", "total sulfur dioxide",
         "density", "pH", "sulphates", "alcohol"], axis=1)
    # creating the tensorflow datasets
    training_dataset = tf.data.Dataset.from_tensor_slices((wine_data_training_input, wine_data_training_label))
    test_dataset = tf.data.Dataset.from_tensor_slices((wine_data_test_input, wine_data_test_label))
    validation_dataset = tf.data.Dataset.from_tensor_slices((wine_data_validation_input, wine_data_validation_label))
    # # apply the data pipeline to the dataset
    prepared_test_data = test_dataset.apply(prepare_wine_data)
    prepared_train_data = training_dataset.apply(prepare_wine_data)
    prepared_validation_data = validation_dataset.apply(prepare_wine_data)
    # initialize the model; one input layer with 11 inputs
    model = MyModel()
    # clean the session
    tf.keras.backend.clear_session()
    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.1
    # Initialize the loss: categorical cross entropy.
    cross_entropy_loss = tf.keras.losses.BinaryCrossentropy()
    # Initialize the optimizer: SGD with default parameters.
    #optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []
    # testing once before we begin
    test_loss, test_accuracy = test(model, prepared_test_data, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    # check how model performs on train data once before we begin
    train_loss, _ = test(model, prepared_train_data, cross_entropy_loss)
    train_losses.append(train_loss)
    #   We train for num_epochs epochs.
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in prepared_train_data:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, prepared_test_data, cross_entropy_loss)
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