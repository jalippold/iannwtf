from tensorflow.python.eager.context import num_gpus
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from denseNet import DenseNet
from resblockmodel import MyResModel
import datetime


def prepare_cifar(cifar10):
    # convert data from uint8 to float32
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    cifar10 = cifar10.map(lambda img, target: (tf.image.per_image_standardization(img), target))
    # create one-hot targets
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    cifar10 = cifar10.cache()
    # shuffle, batch, prefetch
    cifar10 = cifar10.shuffle(1000)
    cifar10 = cifar10.batch(64)
    cifar10 = cifar10.prefetch(32)
    # return preprocessed dataset
    return cifar10

@tf.function
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
        sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_accuracy_aggregator.append(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy



if __name__ == "__main__":
    tf.keras.backend.clear_session()
    
    # loading the data set
    train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True)
    train_ds = train_ds.apply(prepare_cifar)
    test_ds = test_ds.apply(prepare_cifar)
    #print(train_ds.take(1))
    # test preprocessing
    #for elem in train_ds:
    #    print(elem)
    #    break
    ### Hyperparameters
    num_epochs = 30
    learning_rate = tf.constant(0.001, dtype=tf.float32)
    # Initialize the loss: categorical cross entropy.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []
    #create the model
    model = MyResModel()
    # model = DenseNet()
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
    # Visualize accuracy and loss for training and test data.
    plt.figure()
    line1, = plt.plot(train_losses, '-x')
    line2, = plt.plot(test_losses, '-+')
    line3, = plt.plot(test_accuracies, '-o')
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1, line2, line3), ("training loss", "test loss", "test accuracy"))
    plt.show()