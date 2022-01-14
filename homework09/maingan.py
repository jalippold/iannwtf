import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from Generator import Generator
from Discriminator import Discriminator
import time
import os
import urllib.request
import matplotlib.pyplot as plt

def download_data():
    category = 'candle'
    if not os.path.isfile(f"npy_files/{category}.npy"):
        print("Start downloading data...")
        #Download files
        # categories = [line.rstrip(b'\n') for line in urllib.request.urlopen('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
        # print(categories[:10])

        # Creates a folder to download the original drawings into.
        # We chose to use the numpy format : 1x784 pixel vectors, with values going from 0 (white) to 255 (black). We reshape them later to 28x28 grids and normalize the pixel intensity to [-1, 1]

        if not os.path.isdir('npy_files'):
            os.mkdir('npy_files')

        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
        urllib.request.urlretrieve(url, f'npy_files/{category}.npy')
        print("Finished downloading data!")
    else:
        print("Data was already downloaded. Proceeding...")


BATCH_SIZE = 32
NOISE_INPUT_DIM = 100
EPOCHS = 5


def prepare_dataset(dataset):
    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    dataset = dataset.cache()
    # convert data from uint8 to float32
    dataset = dataset.map(lambda vector: tf.cast(vector, tf.float32))
    # sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda vector: (vector / 128.) - 1.)
    # reshape tensor
    dataset = dataset.map(lambda vector: tf.reshape(vector, (28, 28)))
    # expand dim
    dataset = dataset.map(lambda vector: tf.expand_dims(vector, axis=-1))
    # shuffle, batch, prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(32)
    return dataset


# compares the output of the discriminator against a ones tensor for real images
# compares the output of the discriminator against a zeros tensor for generated images
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# the loss for the generator
# it compares the decision of the discriminator for
# the fake outputs against a ones tensor
# idea: the the generator generates 'good' images the
# discriminator will classify them as correct e.g. 1
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_INPUT_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        print(f"Shape of generated images: {generated_images}")

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


if __name__ == "__main__":
    # first download data if not done before
    download_data()

    # # images of candels 141545
    # loading the dataset and split into train and test set
    images = np.load('npy_files/candle.npy')
    print(f"Total num of images: {len(images)}")
    train_images = images[:10000]
    test_images = images[10000:15000]

    # maybe a smaller dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images))
    prepared_train_dataset = train_dataset.apply(prepare_dataset)
    prepared_test_dataset = test_dataset.apply(prepare_dataset)
    print(f"Shape of dataset: {prepared_test_dataset}")

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # since the generator and the discriminator are different nets
    # we need two distinct optimizer
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    generator = Generator(input_dim=NOISE_INPUT_DIM)
    discriminator = Discriminator()
    for epoch in range(EPOCHS):
        start = time.time()
        for image_batch in prepared_train_dataset:
            train_step(image_batch)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

        if epoch%5 == 0:
            print(f"After training for {epoch} epochs the generator can produce following picture:")
            noise = tf.random.normal([1, NOISE_INPUT_DIM])
            generated_image = generator(noise, training=False)
            plt.imshow(tf.squeeze(tf.slice(generated_image,[0,0,0,0],[1,28,28,1])), cmap='gray')
            plt.show()

