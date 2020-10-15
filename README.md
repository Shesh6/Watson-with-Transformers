# Watson with Transformers
 Using Tensorflow and the HuggingFace transformer library along with Weights and Biases to classify entailment in text, for the Kaggle competition "Contradictory, My Dear Watson".
 Inspired by Kaggle user rohanrao, I use sklearn's stratified k-fold cross validation to make sure the training samples are balanced over the languages, and I added TPU support to run on cloud instances. I also use the Google Translate library to optionally translate data for transformers that were trained exclusively on English.
 I avoid extending the default dataset with other open entailment datasets to see what performance can be achieved on the task as it was given.
