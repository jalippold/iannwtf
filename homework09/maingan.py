import numpy as np
import tensorflow as tf
from Generator import Generator
from Discriminator import Discriminator
import time
import datetime
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
    dataset = dataset.shuffle(50000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(32)
    return dataset



BATCH_SIZE = 256
NOISE_INPUT_DIM = 100
EPOCHS = 50


# load tensorboard extension
# %load_ext tensorboard # should be uncommente when working in colab

# Define where to save the log
hyperparameter_string= "Your_Settings_Here"
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# train_log_path = f"test_logs/{hyperparameter_string}/{current_time}/train"
# val_log_path = f"test_logs/{hyperparameter_string}/{current_time}/val"
train_log_path = "test_logs/train"
val_log_path =  "test_logs/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)
# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)


# first download data if not done before
download_data()

# # images of candels 141545
# loading the dataset and split into train and test set
images = np.load('npy_files/candle.npy')
print(f"Total num of images: {len(images)}")
train_images = images[:100000]
test_images = images[100000:140000]

# maybe a smaller dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images))
prepared_train_dataset = train_dataset.apply(prepare_dataset)
prepared_test_dataset = test_dataset.apply(prepare_dataset)
print(f"Shape of dataset: {prepared_test_dataset}")

generator = Generator(input_dim=NOISE_INPUT_DIM)
discriminator = Discriminator()
test_image_noise = tf.random.normal([1, NOISE_INPUT_DIM])

for epoch in range(EPOCHS):
    start = time.time()
    
    # Training:
    for image_batch in prepared_train_dataset:
        noise = tf.random.normal([BATCH_SIZE, NOISE_INPUT_DIM])
        gen_metrics, generated_images = generator.train_step(noise, discriminator)
        disc_metrics =  discriminator.train_step(image_batch, generated_images)

    # print the metrics
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    print([f"gen_{key}: {value}" for (key, value) in zip(list(gen_metrics.keys()), list(gen_metrics.values()))])
    print([f"disc_{key}: {value}" for (key, value) in zip(list(disc_metrics.keys()), list(disc_metrics.values()))])
    
    # logging the training metrics to the log file which is used by tensorboard
    with train_summary_writer.as_default():
        for metric in generator.metrics:
            tf.summary.scalar(f"gen_{metric.name}", metric.result(), step=epoch)
        for metric in discriminator.metrics:
            tf.summary.scalar(f"disc_{metric.name}", metric.result(), step=epoch)
    
    # reset all metrics (requires a reset_metrics method in the model)
    generator.reset_metrics()
    discriminator.reset_metrics()

    # Validation:

    for data in prepared_test_dataset:
        noise = tf.random.normal([BATCH_SIZE, NOISE_INPUT_DIM])
        gen_metrics, generated_images = generator.test_step(noise, discriminator)
        disc_metrics =  discriminator.test_step(image_batch, generated_images)
    
    print([f"gen_val_{key}: {value}" for (key, value) in zip(list(gen_metrics.keys()), list(gen_metrics.values()))])
    print([f"disc_val_{key}: {value}" for (key, value) in zip(list(disc_metrics.keys()), list(disc_metrics.values()))])
    
    # logging the validation metrics to the log file which is used by tensorboard
    with val_summary_writer.as_default():
        for metric in generator.metrics:
            tf.summary.scalar(f"gen_val_{metric.name}", metric.result(), step=epoch)
        for metric in discriminator.metrics:
            tf.summary.scalar(f"disc_val_{metric.name}", metric.result(), step=epoch)

        # every 5 epochs generate images for the test_image_noise and write it to tensorboard
        if (epoch+1)%5 == 0:
            generated_test_images = generator(test_image_noise, training=False)
            # save a batch of images for this epoch (have to be between 0 and 1)
            tf.summary.image(name="generated_images",data = generated_test_images, step=epoch, max_outputs=32)
            
            print("Picture from epoch {} ".format(epoch))
            plt.imshow(generated_images[0, :, :, 0], cmap='gray')
            plt.show()
    
    # reset all metrics
    generator.reset_metrics()
    discriminator.reset_metrics()   
     
    print("\n")

# open the tensorboard to inspect the data for the 100 steps
# %tensorboard --logdir test_logs/train # should be uncommente when working in colab
# %tensorboard --logdir test_logs/val # should be uncommente when working in colab