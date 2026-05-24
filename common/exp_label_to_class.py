from exp.classes.linear import LinearExperiment
from exp.classes.tap import TapExperiment
from exp.classes.renellm import ReNeLLMExperiment
from exp.classes.gptfuzzer import GPTFuzzerExperiment
from exp.classes.pap import PAPExperiment
from exp.classes.utility import UtilityExperiment
from exp.classes.batch_api_exp import BatchAPIExperiment
from exp.classes.batch_api_utility_exp import BatchAPIUtilityExperiment

# Need to handle experiment classes separately to avoid circular imports
# Because experiment classes heavily import other classes
def exp_label_to_class(label: str):
    mapping = {
        "Experiment.LINEAR": LinearExperiment,
        "Experiment.TAP": TapExperiment,
        "Experiment.ReNeLLM": ReNeLLMExperiment,
        "Experiment.GPTFuzzer": GPTFuzzerExperiment,
        "Experiment.PAP": PAPExperiment,
        "Experiment.UTILITY": UtilityExperiment,
        "Experiment.BATCH_API": BatchAPIExperiment,
        "Experiment.BATCH_API_UTILITY": BatchAPIUtilityExperiment,
    }
    return mapping.get(label, None)
