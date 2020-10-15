import tensorflow as tf
from googletrans import Translator

def translate_text(text):
    return Translator().translate(text, dest = "en").text


def encode_text(df, tokenizer, max_len, padding):

    text = df[["premise", "hypothesis"]].values.tolist()

    text_encoded = tokenizer.batch_encode_plus(
        text,
        padding = padding,
        max_length = max_len,
        truncation = True
    )

    return text_encoded


def to_tfds(X, y, auto, labelled = True, repeat = False, shuffle = False, batch_size = 128):

    if labelled:
        ds = (tf.data.Dataset.from_tensor_slices((X["input_ids"], y)))
    else:
        ds = (tf.data.Dataset.from_tensor_slices(X["input_ids"]))

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(2048)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(auto)

    return ds