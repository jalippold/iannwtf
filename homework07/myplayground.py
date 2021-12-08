import tensorflow as tf
from tensorflow.python.framework.tensor_spec import TensorSpec
import numpy as np
import matplotlib.pyplot as plt
import datetime
import argparse

SEQ_LEN = 3
NUM_SAMPLES = 3

MIN_VAL = -1
MAX_VAL = 1

def integration_task(seq_len, num_samples):
    for i in range(num_samples):
        input = tf.random.uniform(shape=(seq_len,), minval=MIN_VAL, maxval=MAX_VAL, dtype=tf.float32)
        target = 1 if tf.math.reduce_sum(input, axis=-1) > 0 else 0
        yield (input, tf.constant(target, dtype=tf.float32))


def my_integration_task():
    for data in integration_task(SEQ_LEN, NUM_SAMPLES):
        yield data


ds = tf.data.Dataset.from_generator(my_integration_task,
            output_signature=(tf.TensorSpec(shape=(SEQ_LEN,), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)))


################################
# Building Dataset done!
################################



def prepare_myds(myds):
    # we don't have to do that much, we build the dataset like we want it...

    # cache this progress in memory, as there is no need to redo it; it is deterministic after all
    myds = myds.cache()
    # shuffle, batch, prefetch
    myds = myds.shuffle(1000)
    myds = myds.batch(64)
    myds = myds.prefetch(32)
    # return preprocessed dataset
    return myds

# loading the data set
train_ds = tf.data.Dataset.from_generator(my_integration_task,
                                              output_signature=(tf.TensorSpec(shape=(SEQ_LEN,), dtype=tf.float32),
                                                                tf.TensorSpec(shape=(), dtype=tf.float32)))
test_ds = tf.data.Dataset.from_generator(my_integration_task,
                                             output_signature=(tf.TensorSpec(shape=(SEQ_LEN,), dtype=tf.float32),
                                                               tf.TensorSpec(shape=(), dtype=tf.float32)))

train_ds = train_ds.apply(prepare_myds)
test_ds = test_ds.apply(prepare_myds)

for elem in train_ds:
    print(elem)