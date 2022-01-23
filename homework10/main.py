from math import ceil
from multiprocessing.spawn import prepare
import tensorflow as tf
import tensorflow_text as tf_txt
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import tqdm
import string

from skipGramModel import SkipGramModel
import pprint


def prepare_dataset(dataset):

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size= tf.data.AUTOTUNE)
    return dataset

### Took from https://www.tensorflow.org/tutorials/text/word2vec ####
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for vocab_size tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
    #   context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)

    #   negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    #       true_classes=context_class,
    #       num_true=1,
    #       num_sampled=num_ns,
    #       unique=True,
    #       range_max=vocab_size,
    #       seed=seed,
    #       name="negative_sampling")

    #   # Build context and label vectors (for one target word)
    #   negative_sampling_candidates = tf.expand_dims(
    #       negative_sampling_candidates, 1)

    #   context = tf.concat([context_class, negative_sampling_candidates], 0)
    #   label = tf.constant([1] + [0]*num_ns, dtype="int64")

    #   # Append each element from the training example to global lists.
    #   targets.append(target_word)
    #   contexts.append(context)
    #   labels.append(label)

        targets.append(target_word)
        contexts.append(context_word)
        labels.append(tf.constant(1, dtype=tf.int64))

  return targets, contexts, labels



BATCH_SIZE = 1024
EPOCHS = 10
VOCAB_SIZE = 10000
WINDOW_SIZE = 2
SEED = 42
NUM_NEG_SAMPLES = 4
SEQ_LENGTH = 10


# load tensorboard extension
# %load_ext tensorboard # should be uncommente when working in colab

train_log_path = "test_logs/train"
val_log_path =  "test_logs/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)
# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)

# prepare dataset
text_ds = tf.data.TextLineDataset("./bible.txt").filter(lambda x: tf.cast(tf.strings.length(x), bool))

# Now, create a custom standardization function to lowercase the text and
# remove punctuation.
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s\d]' % re.escape(string.punctuation), '')

# Define the number of words in a sequence.
vocab_size = VOCAB_SIZE
sequence_length = SEQ_LENGTH

# Use the TextVectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.
vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))

# save vocabulary for reference
inverse_vocab = vectorize_layer.get_vocabulary()
vocab = {word : i for i, word in enumerate(inverse_vocab)}

# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()

sequences = list(text_vector_ds.as_numpy_iterator())


targets, contexts, labels = generate_training_data(sequences, WINDOW_SIZE, NUM_NEG_SAMPLES, vocab_size, SEED)

targets = np.array(targets)
contexts = np.array(contexts)#[:,:,0]
labels = np.array(labels)

print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.apply(prepare_dataset)
print(dataset)

model = SkipGramModel(vocab_sz=VOCAB_SIZE, embed_sz=64)


# eval_tokens = [i for i in range(100, 110)]
# eval_words = [inverse_vocab[token] for token in eval_tokens]
eval_words = ["holy", "father", "wine", "poison", "love", "strong", "day"]
eval_tokens = [vocab[word] for word in eval_words]
pp = pprint.PrettyPrinter(indent=4)
print(f"Eval words and tokens:\n{eval_tokens}\n{eval_words}")

for epoch in range(EPOCHS):
    start = time.time()
    
    # Training:
    for (target, context), label in dataset:    
        metrics = model.train_step(target, context, label)


    # print the metrics
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    
    # logging the training metrics to the log file which is used by tensorboard
    with train_summary_writer.as_default():
        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
    
    # reset all metrics (requires a reset_metrics method in the model)
    model.reset_metrics()

    print(f"Getting nearest neighbours for some example words")
    neigbours = {}
    for token, word in zip(eval_tokens, eval_words):
        nns = model.calculate_nearest_neighbors(token, 5)
        nbs = [inverse_vocab[i] for (i, val) in nns]
        neigbours[word] = nbs

    print("Some words neigbours in the embedding:")
    pp.pprint(neigbours)
    

# open the tensorboard to inspect the data for the 100 steps
# %tensorboard --logdir test_logs/train # should be uncommente when working in colab
# %tensorboard --logdir test_logs/val # should be uncommente when working in colab