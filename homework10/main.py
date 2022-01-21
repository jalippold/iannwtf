from math import ceil
import tensorflow as tf
import tensorflow_text as tf_txt
import numpy as np
import matplotlib.pyplot as plt
import time
import re


def create_input_target_pairs(tokens, window_size=4):
    data = []
    for i, input in enumerate(tokens):
        for j in range(0, ceil(window_size/2)):
            if i-1-j >= 0:
                data.append((input, tokens[i-1-j]))
            if i+1+j < VOCAB_SIZE:
                data.append((input, tokens[i+1+j]))
    return data

def prepare_dataset(dataset):

    # shuffle, batch, prefetch
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.prefetch(128)
    return dataset



BATCH_SIZE = 32
EPOCHS = 10
VOCAB_SIZE = 10000


# load tensorboard extension
# %load_ext tensorboard # should be uncommente when working in colab

train_log_path = "test_logs/train"
val_log_path =  "test_logs/val"

# log writer for training metrics
train_summary_writer = tf.summary.create_file_writer(train_log_path)
# log writer for validation metrics
val_summary_writer = tf.summary.create_file_writer(val_log_path)

# TODO: prepare dataset
with open("./bible.txt", "r") as f:
    text = f.read()
    text = text.lower()
    text = re.sub('[^a-zA-Z ]', '', text.replace("\n", " "))

splitter = tf_txt.WhitespaceTokenizer() #tf_txt.RegexSplitter(split_regex=" *")
text_split = splitter.split(text)[:VOCAB_SIZE]

train_dataset = tf.data.Dataset.from_tensors(create_input_target_pairs(text_split.numpy()))
train_dataset_prepared = train_dataset.apply(prepare_dataset)

print(train_dataset.take(1))

exit(1)

# TODO: init model
model = None

for epoch in range(EPOCHS):
    start = time.time()
    
    # Training:


    # print the metrics
    print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    
    # logging the training metrics to the log file which is used by tensorboard
    with train_summary_writer.as_default():
        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
    
    # reset all metrics (requires a reset_metrics method in the model)
    model.reset_metrics()

    # Validation:



    
    # logging the validation metrics to the log file which is used by tensorboard
    with val_summary_writer.as_default():
        for metric in model.metrics:
            tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
    # reset all metrics
    model.reset_metrics()
    

# open the tensorboard to inspect the data for the 100 steps
# %tensorboard --logdir test_logs/train # should be uncommente when working in colab
# %tensorboard --logdir test_logs/val # should be uncommente when working in colab