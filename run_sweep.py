import wandb
from config import Config
from classifier import build_classifier
from train import train_model
import argparse
import yaml

def parse_args():
    """
    Parse arguments:
    - yaml: path to config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yaml", type=str)

    return parser.parse_args()

def wandb_run():
    """
    Run a single run of the sweep.
    """
    print("Run Started")
    print("Initializing WandB")
    wandb.init()
    
    # Construct config object from wandb config
    config_obj = Config(**wandb.config)
    
    # Train model
    print("Training Model")
    train_model(config_obj, is_wandb=True)

def run_sweep(config_yaml, entity, project):
    """
    Set up and run Weights and Biases hyperparameter sweep from config file.
    """"
    
    print("Setting sweep")
    sweep_id = wandb.sweep(yaml.load(config_yaml), entity=entity, project=project)

    print("Setting agent")
    wandb.agent(sweep_id, wandb_run)

if __name__ == '__main__':

    # Parse arguments
    args = parse_args()

    # Run sweep
    run_sweep(**args)
