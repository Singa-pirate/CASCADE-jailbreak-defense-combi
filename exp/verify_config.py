import logging

from common.exp_label_to_class import exp_label_to_class
from common.label_to_class import label_to_class
from nodes.attacks.offline.base import OfflineAttackNode
from nodes.attacks.online.base import OnlineAttackNode
from nodes.defenses.base import DefenseNode
from nodes.evaluators.base import EvaluatorNode
from nodes.llms.base import LLMBase
from data.classes.base import SingletonDatasetBase
from exp.classes.asr_base import AsrExperimentBase
from exp.classes.batch_api_exp import BatchAPIExperiment
from exp.classes.utility import UtilityExperiment
from exp.classes.batch_api_utility_exp import BatchAPIUtilityExperiment

'''
Verify all fields in config file.
For required fields, check existence, type and validity.
For optional fields, if exists, check type and validity.

Returns True if all checks pass -> entire config file is valid and safe to use later.
'''

def verify_config(config:dict):
    # verify experiment name
    if not _verify_exp_name(config):
        return False
    
    # verify experiment type
    if not _verify_experiment_type(config):
        return False
    
    # when there is attack_sets, verify it
    if "attack_sets" in config:
        if not _verify_attack_sets(config):
            return False

    # when there is defense_sets, verify it
    if "defense_sets" in config:
        if not _verify_defense_sets(config):
            return False

    # when there is evaluation_set, verify it
    if "evaluation_set" in config:
        if not _verify_evaluation_set(config["evaluation_set"]):
            return False
    
    # when there is evaluation_set, verify it
    if "optional_configs" in config:
        if not _verify_optional_configs(config):
            return False
    return True

def _verify_exp_name(config:dict):
    if "exp_name" not in config:
        logging.error(f"Check config file: missing required field 'exp_name'.")
        return False
    if not config["exp_name"] or type(config["exp_name"]) is not str:
        logging.error(f"Check config file: 'exp_name' must be a non-empty string.")
        return False
    return True

def _verify_experiment_type(config:dict):
    # experiment type must be a valid type defined in exp_label_to_class
    # linear and tap experiments can be either interactive or batch
    # utility experiments can only be batch
    if "exp_type" not in config or "interactive" not in config:
        logging.error(f"Check config file: missing required field 'exp_type' or 'interactive'.")
        return False
    experiment_class = exp_label_to_class(config["exp_type"])
    if not experiment_class or not issubclass(experiment_class, AsrExperimentBase) and not issubclass(experiment_class, BatchAPIExperiment) \
            and not issubclass(experiment_class, UtilityExperiment) and not issubclass(experiment_class, BatchAPIUtilityExperiment):
        logging.error(f"Check config file: 'exp_type' has invalid value: {config['exp_type']}.")
        return False
    if experiment_class in [UtilityExperiment, BatchAPIExperiment] and config["interactive"]:
        logging.error(f"Check config file: for the exp_type specified, it can only be run in batch mode (interactive=False).")
        return False
    return True

def _verify_attack_sets(config:dict):
    attack_sets = config["attack_sets"]
    if type(attack_sets) is not list or len(attack_sets) == 0:
        logging.error(f"Check config file: 'attack_sets' must be a non-empty list.")
        return []
    for a_set in attack_sets:
        if not _verify_attack_set(a_set):
            return False
    return True

def _verify_attack_set(a_set:dict):
    # verify required fields
    required_fields = ["name", "attack_sequence"]
    for f in required_fields:
        if f not in a_set:
            logging.error(f"Check config file: attack set missing required field: '{f}'")
            return False
    if not a_set["name"]:
        logging.error(f"Check config file: attack set has empty 'name'.")
        return False
    # attack_sequence is a can-be-empty list, all entries are attack types
    attack_sequence = a_set["attack_sequence"]
    if type(attack_sequence) is not list:
        logging.error(f"Check config file: attack set 'attack_sequence' must be a list.")
        return False
    for a in attack_sequence:
        label = _verify_node_config_and_return_label(a)
        if label is None:
            return False
        a_class = label_to_class(label)
        if not a_class or not (issubclass(a_class, OfflineAttackNode) or issubclass(a_class, OnlineAttackNode)):
            logging.error(f"Check config file: attack set 'attack_sequence' contains invalid attack type: {a}.")
            return False
    if "optional_configs" in a_set:
        optional_configs = a_set["optional_configs"]
        if type(optional_configs) is not dict:
            logging.error(f"Check config file: attack set 'optional_configs' must be a dictionary.")
            return False
    return True

def _verify_defense_sets(config:dict):
    if "defense_sets" not in config:
        logging.error(f"Check config file: missing required field 'defense_sets'.")
        return False
    defense_sets = config["defense_sets"]
    if type(defense_sets) is not list or len(defense_sets) == 0:
        logging.error(f"Check config file: 'defense_sets' must be a non-empty list.")
        return False
    for d_set in defense_sets:
        if not _verify_defense_set(d_set):
            return False
    return True

def _verify_defense_set(d_set:dict):
    # verify required fields
    required_fields = ["name", "defense_sequence_before_llm", "target_llm", "defense_sequence_after_llm"]
    for f in required_fields:
        if f not in d_set:
            logging.error(f"Check config file: defense set missing required field: '{f}'")
            return False
    if not d_set["name"]:
        logging.error(f"Check config file: defense set has empty 'name'.")
        return False
    # target_llm is a LLM type
    target_llm_label = _verify_node_config_and_return_label(d_set["target_llm"])
    if target_llm_label is None:
        return False
    target_llm_class = label_to_class(target_llm_label)
    if not target_llm_class or not issubclass(target_llm_class, LLMBase):
        logging.error(f"Check config file: defense set has invalid target_llm: {d_set['target_llm']}.")
        return False
    # defense_sequence_before_llm, defense_sequence_after_llm, are can-be-empty lists, all entries are defense types
    sequences = ["defense_sequence_before_llm", "defense_sequence_after_llm"]
    for seq in sequences:
        if type(d_set[seq]) is not list:
            logging.error(f"Check config file: defense set '{seq}' must be a list.")
            return False
        for d in d_set[seq]:
            d_label = _verify_node_config_and_return_label(d)
            if d_label is None:
                return False
            d_class = label_to_class(d_label)
            if not d_class or not issubclass(d_class, DefenseNode):
                logging.error(f"Check config file: defense set '{seq}' contains invalid defense type: {d}.")
                return False
    if "optional_configs" in d_set:
        optional_configs = d_set["optional_configs"]
        if type(optional_configs) is not dict:
            logging.error(f"Check config file: defense set 'optional_configs' must be a dictionary.")
            return False
    return True

def _verify_evaluation_set(e_set:dict):
    # verify required fields
    required_fields = ["name", "evaluator_sequence", "dataset"]
    for f in required_fields:
        if f not in e_set:
            logging.error(f"Check config file: evaluation set missing required field: '{f}'")
            return False
    if not e_set["name"]:
        logging.error(f"Check config file: evaluation set has empty 'name'.")
        return False
    # evaluator_sequence is a can-be-empty list, all entries are evaluator types
    evaluator_sequence = e_set["evaluator_sequence"]
    if type(evaluator_sequence) is not list:
        logging.error(f"Check config file: evaluation set 'evaluator_sequence' must be a list.")
        return False
    for e in evaluator_sequence:
        e_label = _verify_node_config_and_return_label(e)
        if e_label is None:
            return False
        e_class = label_to_class(e_label)
        if not e_class or not issubclass(e_class, EvaluatorNode):
            logging.error(f"Check config file: evaluation set 'evaluator_sequence' contains invalid evaluator type: {e}.")
            return False
    # dataset is a dataset type
    dataset_label = _verify_node_config_and_return_label(e_set["dataset"])
    if dataset_label is None:
        return False
    dataset_class = label_to_class(dataset_label)
    if not dataset_class or not issubclass(dataset_class, SingletonDatasetBase):
        logging.error(f"Check config file: evaluation set has invalid dataset: {e_set['dataset']}.")
        return False
    if "optional_configs" in e_set:
        optional_configs = e_set["optional_configs"]
        if type(optional_configs) is not dict:
            logging.error(f"Check config file: evaluation set 'optional_configs' must be a dictionary.")
            return False
    return True

def _verify_optional_configs(config:dict):
    optional_configs = config["optional_configs"]
    if type(optional_configs) is not dict:
        logging.error(f"Check config file: 'optional_configs' must be a dictionary.")
        return False
    return True

def _verify_node_config_and_return_label(node_config: str | dict) -> str | None:
    # For all nodes (attack, defense, llm, evaluator, dataset), node_config can be either a string label or a dict with label and optional configs
    # the label string mus be non-empty
    # return the label string if valid, else None
    if type(node_config) is not str and type(node_config) is not dict:
        logging.error(f"Check config file: config for a node must be either string label or dict with label and optional configs.")
        return None
    if type(node_config) is dict:
        # must have 'label' and 'optional_configs' fields
        if "label" not in node_config or "optional_configs" not in node_config:
            logging.error(f"Check config file: config for a node dict must have 'label' and 'optional_configs' fields.")
            return None
        if not _verify_optional_configs(node_config):
            return None
        label = node_config["label"]
    else:
        label = node_config
    if not label or type(label) is not str:
        logging.error(f"Check config file: node label must be a non-empty string.")
        return None
    return label
