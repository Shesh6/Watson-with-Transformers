import pandas as pd
import numpy as np
from config import Config
from classifier import build_classifier
from train import train_model
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str)
    parser.add_argument("-m", "--model_save", type=str)
    parser.add_argument("-p", "--preds_save", type=str)
    return parser.parse_args()

def run(args):
    print(args.yaml)
    config_dict = yaml.load(open(args.yaml))
    print(config_dict)
    config_obj = Config(**config_dict)
    print("Configuration object set")
    model, preds_oof, preds_test = train_model(config_obj, is_wandb=False)
    df_test = pd.read_csv(config_obj.PATH_TEST)
    df_submission = pd.DataFrame({"id": df_test.id.values, "prediction": np.argmax(preds_test, axis = 1)})
    df_submission.to_csv(args.preds_save+"/submission.csv", index = False)
    model.save(args.model_save)

if __name__ == '__main__':
    args = parse_args()
    run(args)
