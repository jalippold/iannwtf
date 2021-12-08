import tensorflow as tf
import tensorflow_datasets as tfds
SEQ_LEN = 10
NUM_SAMPLES = 6400

MIN_VAL = -1
MAX_VAL = 1

def integration_task(seq_len, num_samples):
    for i in range(num_samples):
        input = tf.random.uniform(shape=(seq_len,1), minval=MIN_VAL, maxval=MAX_VAL, dtype=tf.float32)
        target = 1 if tf.math.reduce_sum(input, axis=0) > 0 else 0
        yield (input, tf.constant(target, dtype=tf.float32, shape = (1)))


def my_integration_task():
    for data in integration_task(SEQ_LEN, NUM_SAMPLES):
        yield data

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



test_ds = tf.data.Dataset.from_generator(my_integration_task,
                                         output_signature=(tf.TensorSpec(shape=(SEQ_LEN,1), dtype=tf.float32),
                                                          tf.TensorSpec(shape=(1), dtype=tf.float32))).batch(64)

#test_ds.apply(prepare_myds)
for (input, target) in test_ds:
    print(input)