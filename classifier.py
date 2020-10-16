import tensorflow as tf
from transformers import TFAutoModel

def build_classifier(model_name, max_len, learning_rate, metrics):
    """
    Constructing a transformer model given a configuration.
    """
    # Defining the encoded inputs
    input_ids = tf.keras.layers.Input(shape = (max_len,), dtype = tf.int32, name = "input_ids")
    
    # Loading pretrained transformer model
    transformer_model = TFAutoModel.from_pretrained(model_name)

    # Defining the data embedding using the loaded model
    transformer_embeddings = transformer_model(input_ids)[0]

    # Defining the classifier layer
    output_values = tf.keras.layers.Dense(3, activation = "softmax")(transformer_embeddings[:, 0, :])

    # Constructing the final model along with an optimizer, loss function and metrics
    model = tf.keras.Model(inputs = input_ids, outputs = output_values)
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    metrics = metrics
    model.compile(optimizer = opt, loss = loss, metrics = metrics)

    return model