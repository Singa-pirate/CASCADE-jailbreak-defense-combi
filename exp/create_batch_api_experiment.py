import logging, os

from common.exp_label_to_class import exp_label_to_class

def create_batch_api_experiment(config:dict, config_file:str):
    exp_name = config["exp_name"]
    exp_type = config["exp_type"]
    exp_class = exp_label_to_class(exp_type)
    
    optional_configs = config.get("optional_configs", {})

    valid_attack_sets = config.get("attack_sets", None) # note: already verified
    valid_defense_sets = config.get("defense_sets", None)
    evaluation_set = config.get("evaluation_set", None)

    logging.basicConfig(level=logging.ERROR)

    if valid_attack_sets is None or len(valid_attack_sets) == 0:
        logging.error("Batch API experiment requires at least one valid attack set.")
        return None
    if valid_defense_sets is None or len(valid_defense_sets) == 0:
        logging.error("Batch API experiment requires at least one valid defense set.")
        return None
    if evaluation_set is None:
        logging.error("Batch API experiment requires a valid evaluation set.")
        return None
        
    save_dir = _create_batch_experiment_output_dir(exp_name, config_file)
    if save_dir is None:
        return None
    
    exp = exp_class(
        exp_name=exp_name,
        attack_sets=valid_attack_sets,
        defense_sets=valid_defense_sets,
        evaluation_set=evaluation_set,
        save_dir=save_dir,
        optional_configs=optional_configs,
    )
    return exp

def _create_batch_experiment_output_dir(exp_name:str, config_file:str):
        # exp name must be valid directory name, for creating folder to save results
        valid_dir_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in exp_name:
            if c not in valid_dir_chars:
                logging.error(f"Check config file: 'exp_name' contains invalid character '{c}'. Only alphanumeric, - and _ are allowed.")
                return None
        
        # exp name should not be existing directory
        save_dir = f"outputs/results/asr/{exp_name}"
        if os.path.exists(save_dir) and os.path.isdir(save_dir):
            files = os.listdir(save_dir)
            result_files = [f for f in files if f.endswith('.parquet') and 'LLM_outputs' not in f and 'before_target_LLM' not in f]
            if len(result_files) > 0:
                logging.error(f"Check config file: 'exp_name' already has results in outputs/results/asr. Please choose another name.")
                return None

        os.makedirs(save_dir, exist_ok=True)

        # copy over config file
        with open(config_file, 'r') as f:
            config_file_content = f.read()
            f.close()
        with open(f"{save_dir}/config.yaml", 'w') as f:
            f.write(config_file_content)
            f.close()

        return save_dir