from fileinput import filename
import tensorflow as tf
import tensorflow_text as tf_txt
import io
import datetime
import tqdm
import os
import sentencepiece as sp

VOCAB_SIZE = 2000           # something between 2000 and 7000
SLIDING_WINDOW_SIZE = 128    # somethin between 32 and 256
BATCH_SIZE = 256


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
    dataset = tf_txt.sliding_window(tokens, SLIDING_WINDOW_SIZE+1)      # exercise sheet says to do this after building the dataset from tokens but that raises error for me...
    dataset = tf.data.Dataset.from_tensor_slices(tokens)
    dataset = dataset.apply(prepare_dataset)

    print(dataset)

