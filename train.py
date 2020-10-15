import tensorflow as tf
import pandas as pd
import numpy as np
import wandb
import gc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from utils import translate_text, encode_text, to_tfds
from classifier import build_classifier

def train_model(config, is_wandb = False):

    print("reading data")
    df_train = pd.read_csv(config.PATH_TRAIN)
    df_test = pd.read_csv(config.PATH_TEST)

    if is_wandb:    
        wb = wandb.keras.WandbCallback()
    
    print("translating data")
    if config.TRANSLATION:
        df_train.loc[df_train.language != "English", "premise"] = df_train[df_train.language != "English"].premise.apply(lambda x: translate_text(x))
        df_test.loc[df_test.language != "English", "premise"] = df_test[df_test.language != "English"].premise.apply(lambda x: translate_text(x))

        df_train.loc[df_train.language != "English", "hypothesis"] = df_train[df_train.language != "English"].hypothesis.apply(lambda x: translate_text(x))
        df_test.loc[df_test.language != "English", "hypothesis"] = df_test[df_test.language != "English"].hypothesis.apply(lambda x: translate_text(x))

    # adding column for stratified splitting
    df_train["language_label"] = df_train.language.astype(str) + "_" + df_train.label.astype(str)

    # stratified K-fold on language and label
    skf = StratifiedKFold(n_splits = config.TRAIN_SPLITS, shuffle = True, random_state = config.SEED)

    print("initializing predictions")
    preds_oof = np.zeros((df_train.shape[0], 3))
    preds_test = np.zeros((df_test.shape[0], 3))
    acc_oof = []

    print("iterating over folds")
    for (fold, (train_index, valid_index)) in enumerate(skf.split(df_train, df_train.language_label)):
        # initializing TPU
        if config.ACCELERATOR == "TPU":
            if config.tpu:
                config.initialize_accelerator()

        print("building model")
        tf.keras.backend.clear_session()
        with config.strategy.scope():
            model = build_classifier(config.MODEL_NAME, config.MAX_LENGTH, config.LEARNING_RATE, config.METRICS)
            if fold == 0:
                print(model.summary())

        print("\n")
        print("#" * 19)
        print(f"##### Fold: {fold + 1} #####")
        print("#" * 19)

        # splitting data into training and validation
        X_train = df_train.iloc[train_index]
        X_valid = df_train.iloc[valid_index]

        y_train = X_train.label.values
        y_valid = X_valid.label.values

        print("\nTokenizing")

        # encoding text data using tokenizer
        X_train_encoded = encode_text(df = X_train, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)
        X_valid_encoded = encode_text(df = X_valid, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)

        # creating TF Dataset
        ds_train = to_tfds(X_train_encoded, y_train, config.AUTO, repeat = True, shuffle = True, batch_size = config.BATCH_SIZE * config.REPLICAS)
        ds_valid = to_tfds(X_valid_encoded, y_valid, config.AUTO, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)

        n_train = X_train.shape[0]

        if fold == 0:
            X_test_encoded = encode_text(df = df_test, tokenizer = config.TOKENIZER, max_len = config.MAX_LENGTH, padding = config.PAD_TO_MAX_LENGTH)

        # saving model at best accuracy epoch
        sv = tf.keras.callbacks.ModelCheckpoint(
            "models\model.h5",
            monitor = "val_sparse_categorical_accuracy",
            verbose = 0,
            save_best_only = True,
            save_weights_only = True,
            mode = "max",
            save_freq = "epoch"
        )

        cbs = [sv]
        if is_wandb:
            cbs.append(wb)
        
        print("\nTraining")

        # training model
        model_history = model.fit(
            ds_train,
            epochs = config.EPOCHS,
            callbacks = cbs,
            steps_per_epoch = n_train / config.BATCH_SIZE // config.REPLICAS,
            validation_data = ds_valid,
            verbose = config.VERBOSE
        )

        print("\nValidating")

        # scoring validation data
        model.load_weights("models\model.h5")
        ds_valid = to_tfds(X_valid_encoded, -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)

        preds_valid = model.predict(ds_valid, verbose = config.VERBOSE)
        acc = accuracy_score(y_valid, np.argmax(preds_valid, axis = 1))

        preds_oof[valid_index] = preds_valid
        acc_oof.append(acc)

        print("\nInferencing")

        # scoring test data
        ds_test = to_tfds(X_test_encoded, -1, config.AUTO, labelled = False, batch_size = config.BATCH_SIZE * config.REPLICAS * 4)
        preds_test += model.predict(ds_test, verbose = config.VERBOSE) / config.TRAIN_SPLITS

        print(f"\nFold {fold + 1} Accuracy: {round(acc, 4)}\n")

        g = gc.collect()

    # overall CV score and standard deviation
    print(f"\nCV Mean Accuracy: {round(np.mean(acc_oof), 4)}")
    print(f"CV StdDev Accuracy: {round(np.std(acc_oof), 4)}\n")

    return model, preds_oof, preds_test