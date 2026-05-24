import yaml

from common.exp_label_to_class import exp_label_to_class
from exp.classes.asr_base import AsrExperimentBase 
from exp.classes.batch_api_exp import BatchAPIExperiment
from exp.classes.utility import UtilityExperiment

from .verify_config import verify_config
from .create_asr_experiment import create_asr_experiment
from .create_batch_api_experiment import create_batch_api_experiment
from .create_utility_experiment import create_utility_experiment
from .create_batch_api_utility_experiment import create_batch_api_utility_experiment

def run_experiment(config_file='config.yaml'):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if not verify_config(config):
        raise Exception(f"Invalid config file format: {config_file}")

    exp_class = exp_label_to_class(config["exp_type"])
    if issubclass(exp_class, AsrExperimentBase):
        exp = create_asr_experiment(config, config_file)
    elif issubclass(exp_class, BatchAPIExperiment):
        exp = create_batch_api_experiment(config, config_file)
    elif issubclass(exp_class, UtilityExperiment):
        exp = create_utility_experiment(config, config_file)
    else:
        exp = create_batch_api_utility_experiment(config, config_file)
    
    if exp is None:
        raise Exception("Failed to create valid experiment. Check config.")

    exp.run()
