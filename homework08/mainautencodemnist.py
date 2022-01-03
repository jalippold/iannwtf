import tensorflow as tf
from tensorflow.python.framework.tensor_spec import TensorSpec
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow_datasets as tfds
from sklearn.manifold import TSNE
from scipy.interpolate import interp1d

def prepare_mnist(dataset):
    # we don't have to do that much, we build the dataset like we want it...

    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    dataset = dataset.cache()
    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # change target to original image
    dataset = dataset.map(lambda img, target: (img, img))
    # add dimension, nicht notwendig hat schon entsprechende Dimension
    #dataset = dataset.map(lambda img, target: (tf.expand_dims(img,axis=-1), tf.expand_dims(target,axis=-1)))
    # clip values of input image
    #dataset = dataset.map(lambda img, target: (tf.clip_by_value(img,clip_value_min=0,clip_value_max=255), target))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))
    # add noise
    multiply_factor = 0.2
    dataset = dataset.map(lambda img, target: (tf.math.add(img,tf.multiply(multiply_factor,tf.random.normal(shape=img.shape))), target))
    # clip values of input image
    dataset = dataset.map(lambda img, target: (tf.clip_by_value(img,clip_value_min=-1,clip_value_max=1), target))
    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(64, drop_remainder=True)
    dataset = dataset.prefetch(32)
    # return preprocessed dataset
    return dataset

def prepare_for_embedding_test(dataset):
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    dataset = dataset.cache()
    # convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img / 128.) - 1., target))
    # add noise
    multiply_factor = 0.2
    dataset = dataset.map(lambda img, target: (tf.math.add(img, tf.multiply(multiply_factor, tf.random.normal(shape=img.shape))), target))
    # clip values of input image
    dataset = dataset.map(lambda img, target: (tf.clip_by_value(img, clip_value_min=-1, clip_value_max=1), target))
    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    #for evaluation no batch needed
    dataset = dataset.batch(64, drop_remainder=True)
    dataset = dataset.prefetch(32)
    # return preprocessed dataset
    return dataset



@tf.function
def train_step(model, input, target,loss_function,optimizer):
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


def test(model, test_data, loss_fuction):
    """
    # Tests over complete test data
    :param model: the network model
    :param test_data: the given input tensors
    :param loss_function: the given loss function
    :return: the test loss and test accuracy
    """
    counter = 0
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (input, target) in test_data:
        # print(input)
        prediction = model(input, training=False)
        if (counter < 3):
          fig = plt.figure
          print("input image")
          plt.imshow(tf.squeeze(tf.slice(input,[0,0,0,0],[1,28,28,1])), cmap='gray')
          plt.show()
          print("prediction image")
          plt.imshow(tf.squeeze(tf.slice(prediction,[0,0,0,0],[1,28,28,1])), cmap='gray')
          plt.show()
          counter += 1
        sample_test_loss = loss_fuction(target, prediction)
        sample_test_accuracy = target == np.round(prediction)
        # print(sample_test_accuracy)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_accuracy_aggregator.append(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)
    return test_loss, test_accuracy


if __name__ == "__main__":

    tf.keras.backend.clear_session()

    # loading the data set
    train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

    train_ds_prepared = train_ds.apply(prepare_mnist)
    test_ds_prepared = test_ds.apply(prepare_mnist)
    test_ds_embedding = test_ds.apply(prepare_for_embedding_test)
    # show input when needed
    #for x,y in train_ds:
    #   plt.imshow(tf.squeeze(tf.slice(x,[0,0,0,0],[1,28,28,1])), cmap='gray')
    #   plt.imshow(tf.squeeze(tf.slice(y,[0,0,0,0],[1,28,28,1])), cmap='gray')
    #   plt.show()
    ### Hyperparameters
    num_epochs = 10
    learning_rate = tf.constant(0.001, dtype=tf.float32)
    # Initialize the loss: total variation to measure the noise
    loss_function = tf.keras.losses.MeanSquaredError()
    # Initialize the optimizer: ADAM with default parameters.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    #create the model
    model = AutoEncoderModel()
    # testing once before we begin
    test_loss, test_accuracy = test(model, test_ds_prepared,loss_function)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    # check how model performs on train data once before we begin
    train_loss, _ = test(model, train_ds_prepared,loss_function)
    train_losses.append(train_loss)
    #check embedding
    myvector = tf.random.uniform(shape=(1,28,28,1), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
    print(model.encoder(myvector))
    #   We train for num_epochs epochs.
    for epoch in range(num_epochs):
        start_time = datetime.datetime.now()
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in train_ds_prepared:
            train_loss = train_step(model, input, target,loss_function, optimizer)
            epoch_loss_agg.append(train_loss)
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_ds_prepared,loss_function)
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
    print("---------------- Training finished -------------------------------")
    print("---------------- Looking at the embedding ------------------------")
    counter_embedding = 0
    embedding_list = []
    embedding_target_list =[]
    for (input,target) in test_ds_embedding:
        # think of batching
        if counter_embedding < 16:
            embedding_list.append(model.encoder(input))
            embedding_target_list.append(target)
        else:
            break
        counter_embedding += 1
    # convert batched tensor into normal tensor
    single_embedding_list = []
    single_label_list = []
    for i in range(0,len(embedding_list)):
        for j in range(0,64):
            #print(embedding_list[i])
            #print(embedding_list[i].shape)
            current_slice = tf.slice(embedding_list[i], [j,0], [1,10])
            single_embedding_list.append(tf.squeeze(current_slice))
            single_label_list.append(tf.slice(embedding_target_list[i], [j], [1]))
    print(len(single_embedding_list))
    print(len(single_label_list))
    array_embedding_list = np.asarray(single_embedding_list)
    #print(array_embedding_list)
    print(array_embedding_list.shape)
    X_embedded = TSNE(n_components=2, learning_rate='auto',init = 'random').fit_transform(array_embedding_list)
    # find right color for each point
    colors = ["blue", "brown", "green","red","pink","gray","olive","cyan","orange","purple"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    firstimage_embedding = array_embedding_list[0]
    secondimage_embedding =  array_embedding_list[1]
    reconstructed_image1 = model.decoder(tf.expand_dims(firstimage_embedding, axis=0))
    reconstructed_image2 = model.decoder(tf.expand_dims(secondimage_embedding, axis=0))
    f = interp1d(firstimage_embedding, secondimage_embedding, kind='linear', fill_value="extrapolate")
    print(f)
    x = np.linspace(min(firstimage_embedding + secondimage_embedding), max(firstimage_embedding + secondimage_embedding ), num=100, endpoint=True)
    for i in range(len(X_embedded)):
        #print(X_embedded[i][0])
        #print(X_embedded[i][1])
        #print(single_label_list[i].numpy()[0])
        #print(colors[single_label_list[i].numpy()[0]])
        ax.scatter(X_embedded[i][0], X_embedded[i][1], color=colors[single_label_list[i].numpy()[0]])
    plt.plot(x,f(x), 'mx')
    plt.show()
    for i in range(0,len(colors)):
      print('{:d} : {:s}'.format(i, colors[i]))
    print('------------------just interpolation plot to detect it------------------------------------------')
    plt.plot(x,f(x), 'mx')
    plt.show()
    print('------------------interpolation images------------------------------------------')
    print("first reconstructed image")
    plt.imshow(tf.squeeze(tf.slice(reconstructed_image1,[0,0,0,0],[1,28,28,1])), cmap='gray')
    plt.show()
    print("second reconstructed image")
    plt.imshow(tf.squeeze(tf.slice(reconstructed_image2,[0,0,0,0],[1,28,28,1])), cmap='gray')
    plt.show()