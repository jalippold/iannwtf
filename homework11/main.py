from MyNLPModel import MyNLPModel

import tensorflow as tf
import tensorflow_text as tf_txt
import io
import datetime
import tqdm
import time
import sentencepiece as sp

VOCAB_SIZE = 2000           # something between 2000 and 7000
SLIDING_WINDOW_SIZE = 128   # somethin between 32 and 256
BATCH_SIZE = 256
EMBED_DIM = 128             # something between 64 and 256
DENSE_DIM = 64              # someting between 32 and 256
EPOCHS = 10


def get_data():
    filename = "nietzsche"
    path = tf.keras.utils.get_file("./nietzsche.txt", origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")

    with open(path, "r") as f:
        text = f.read()

    sp.SentencePieceTrainer.train(
        input=path, model_prefix=f'{filename}_tokenizer_model', model_type="unigram", vocab_size=VOCAB_SIZE)

    # deserialize the trained model file to load it in the correct format
    trained_tokenizer_model = tf.io.gfile.GFile(f'{filename}_tokenizer_model.model', "rb").read()

    # load the model as a tokenizer that can be used inside a tensorflow model
    tokenizer = tf_txt.SentencepieceTokenizer(
        model=trained_tokenizer_model, out_type=tf.int32, nbest_size=-1, alpha=1, reverse=False,
        add_bos=False, add_eos=False, return_nbest=False, name=None
    )
    return tokenizer, text

def prepare_dataset(dataset):
    # shuffle, batch, prefetch
    dataset = dataset.shuffle(5000)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.cache()
    dataset = dataset.prefetch(buffer_size= tf.data.AUTOTUNE)
    return dataset





if __name__ == "__main__":

    tf.keras.backend.clear_session()

    ### dataset, preprocessing and tokenization
    tokenizer, text = get_data()
    tokens = tokenizer.tokenize(text)
    dataset = tf_txt.sliding_window(tokens, SLIDING_WINDOW_SIZE+1)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.map(lambda x: tf.split(x, [SLIDING_WINDOW_SIZE, 1], axis=0, num=2))
    dataset = dataset.apply(prepare_dataset)

    model = MyNLPModel(tokenizer=tokenizer, seq_len=SLIDING_WINDOW_SIZE, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, dense_dim=DENSE_DIM)

    ### Set up tensorboard
    # load tensorboard extension
    # %load_ext tensorboard

    # Define where to save the log
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_path = f"logs/{current_time}/train"

    # log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)

    ### training loop
    for epoch in range(EPOCHS):
        start = time.time()
    
        print(f"Epoch {epoch}:")
        
        # Training:
        
        # for data in tqdm.notebook.tqdm(dataset, position=0, leave=True):
        for data in dataset:
            metrics = model.train_step(data)
        
        # print the metrics
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print([f"{key}: {value}" for (key, value) in zip(list(metrics.keys()), list(metrics.values()))])
        
        # logging the validation metrics to the log file which is used by tensorboard
        with train_summary_writer.as_default():
            for metric in model.metrics:
                tf.summary.scalar(f"{metric.name}", metric.result(), step=epoch)
        
        # reset all metrics (requires a reset_metrics method in the model)
        model.reset_metrics()
        
        print("\n")

    # open the tensorboard
    # %tensorboard --logdir logs/