import wandb
from config import Config
from classifier import build_classifier
from train import train_model
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str)
    return parser.parse_args()

def wandb_run():
    print("Run Started")
    wandb.init()
    print("WandB initialized")
    config_obj = Config(**wandb.config)
    print("Configuration object set")
    train_model(config_obj, is_wandb=True)

def run_sweep(config_yaml, entity, project):
    print("setting sweep")
    sweep_id = wandb.sweep(yaml.load(config_yaml), entity=entity, project=project)
    print("setting agent")
    wandb.agent(sweep_id, wandb_run)

if __name__ == '__main__':
    args = parse_args()
    run_sweep(**args)
