# Watson with Transformers
 Using Tensorflow and the HuggingFace transformer library along with Weights and Biases to classify entailment in text, for the Kaggle competition "Contradictory, My Dear Watson".

 Inspired by Kaggle user rohanrao, I use sklearn's stratified k-fold cross validation to make sure the training samples are balanced over the languages, and I added TPU support to run on cloud instances. I also use the Google Translate library to optionally translate data for transformers that were trained exclusively on English.

The model reached *29th place* with on Kaggle 92.7% accuracy on the test set when fine-tuned on an XLM RoBERTAa Large transformer model that was pretrained on the XNLI entailment dataset. It was trained in 55 minutes on a single TPU in a free Kaggle instance, using the command in `scripts/train_xnli`, which call on the configuration at `config/config_3`.
