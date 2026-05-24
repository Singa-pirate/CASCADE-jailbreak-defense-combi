import os
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import sys
from exp.run_experiment import run_experiment

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_experiment(config_file=config_file)
