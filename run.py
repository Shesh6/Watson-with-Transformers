import pandas as pd
import numpy as np
from config import Config
from classifier import build_classifier
from train import train_model
import argparse
import yaml

def parse_args():
    """
    Parse arguments:
    - yaml: path to config file.
    - model_save: path to save model.
    - preds_save: path to save predictions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str)
    parser.add_argument("-m", "--model_save", type=str)
    parser.add_argument("-p", "--preds_save", type=str)

    return parser.parse_args()

def run(args):
    """
    Run training program.
    """

    # Load YAML
    config_dict = yaml.load(open(args.yaml))

    # Construct config object
    config_obj = Config(**config_dict)

    # Train model
    model, preds_oof, preds_test = train_model(config_obj, is_wandb=False)

    # Save test predictions
    df_test = pd.read_csv(config_obj.PATH_TEST)
    df_submission = pd.DataFrame({"id": df_test.id.values, "prediction": np.argmax(preds_test, axis = 1)})
    df_submission.to_csv(args.preds_save, index = False)

    # Save model
    model.save(args.model_save)

if __name__ == '__main__':
    
    # Parse arguments
    args = parse_args()

    # Run traning
    run(args)
