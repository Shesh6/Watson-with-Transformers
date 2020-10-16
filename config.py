import tensorflow as tf
from pathlib import Path
from transformers import AutoTokenizer
 
class Config():
    """
    Class for holding the configuration of a run, and setting up the accelerator strategy.
    Arguments:
    - model_name: Title of model from the HuggingFace directory of transformers.
    - translation: Boolean determining whether non-English data should be translated using
                   the Google Translate library. Used for transformers that were trained
                   exclusively on English data.
    - max_length: Maximum ength of sequences processed by the transformer. The longer, the
                  more long-term information can be learned, but takes up more memory.
    - padding: Bollean determining whether tokenizer should pad sequences to max_length.
    - batch_size
    - epochs
    - learning_rate
    - metrics: metrics to be logged.
    - verbose
    - train_splits: how many folds to perform stratified cross-validation with.
    - accelerator: 'TPU' or 'GPU'
    - random_seed
    """


    def __init__(
        self,
        model_name,
        translation = True,
        max_length = 64,
        padding = True,
        batch_size = 128,
        epochs = 5,
        learning_rate = 1e-5,
        metrics = ["sparse_categorical_accuracy"],
        verbose = 1,
        train_splits = 5,
        accelerator = "TPU",
        random_seed = 6
    ):

        # Set up
        self.SEED = random_seed
        self.ACCELERATOR = accelerator
        self.PATH_TRAIN = Path("data/train.csv")
        self.PATH_TEST  = Path("data/test.csv")
        self.TRAIN_SPLITS = train_splits

        # Model configuration
        self.MODEL_NAME = model_name
        self.TRANSLATION = translation
        self.TOKENIZER = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        # Model hyperparameters
        self.MAX_LENGTH = max_length
        self.PAD_TO_MAX_LENGTH = padding
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.LEARNING_RATE = learning_rate
        self.METRICS = metrics
        self.VERBOSE = verbose

        # Language maps
        self.LANGUAGE_MAP = {
            "English"   : 0,
            "Chinese"   : 1,
            "Arabic"    : 2,
            "French"    : 3,
            "Swahili"   : 4,
            "Urdu"      : 5,
            "Vietnamese": 6,
            "Russian"   : 7,
            "Hindi"     : 8,
            "Greek"     : 9,
            "Thai"      : 10,
            "Spanish"   : 11,
            "German"    : 12,
            "Turkish"   : 13,
            "Bulgarian" : 14
        }
        self.INVERSE_LANGUAGE_MAP = {v: k for k, v in self.LANGUAGE_MAP.items()}
        
        # Initializing accelerator
        self.initialize_accelerator()

    def initialize_accelerator(self):
        """
        Method for initializing accelerator strategy (TPU or GPU)
        """
        # Checking TPU first
        if self.ACCELERATOR == "TPU":
            print("Connecting to TPU")
            try:
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
                print(f"Running on TPU {tpu.master()}")
            except ValueError:
                print("Could not connect to TPU")
                tpu = None

            if tpu:
                try:
                    print("Initializing TPU")
                    tf.config.experimental_connect_to_cluster(tpu)
                    tf.tpu.experimental.initialize_tpu_system(tpu)
                    self.strategy = tf.distribute.experimental.TPUStrategy(tpu)
                    self.tpu = tpu
                    print("TPU initialized")
                except _:
                    print("Failed to initialize TPU")
            else:
                print("Unable to initialize TPU")
                self.ACCELERATOR = "GPU"

        # Default for CPU and GPU otherwise
        else:
            print("Using default strategy for CPU and single GPU")
            self.strategy = tf.distribute.get_strategy()

        # Checking GPUs
        if self.ACCELERATOR == "GPU":
            print(f"GPUs Available: {len(tf.config.experimental.list_physical_devices('GPU'))}")

        # Defining replicas
        self.AUTO = tf.data.experimental.AUTOTUNE
        self.REPLICAS = self.strategy.num_replicas_in_sync
        print(f"REPLICAS: {self.REPLICAS}")