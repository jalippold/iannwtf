import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import generator
import Discriminator
import time

BATCH_SIZE = 64


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
    dataset = dataset.batch(64, drop_remainder=True)
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
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


if __name__ == "__main__":
    # # images of candels 141545
    # loading the dataset
    images = np.load('npy_files/candle.npy')
    # print(len(images))
    # print(images)
    # maybe a smaller dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((images))
    print(train_dataset)
    prepared_dataset = train_dataset.apply(prepare_dataset)
    print('hello')
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # since the generator and the discriminator are different nets
    # we need two distinct optimizer
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    num_Epoch = 50
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    generator = generator()
    discriminator = Discriminator()
    for epoch in range(num_Epoch):
        start = time.time()
        for image_batch in prepared_dataset:
            train_step(image_batch)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # print(len(list(train_dataset)))
