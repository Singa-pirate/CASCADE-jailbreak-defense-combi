## Run Utility Experiment

To run utility experiment using AlpacaEval:

1. Install alpaca_eval and patch it ([reference](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_vllm_llama3_70b))
   - After setting up venv, go to alpaca_eval package
   - Go to `evalators_configs/weighted_alpaca_eval_vllm_llama3_70b`, in config.yaml, configure a suitable model_name (quantised version of Llama-3-70B, e.g. `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4`)

2. First, run utility experiment using `script.sh`, with only `generate_only=True` in the config. This will only let the defense graph generate outputs and save. They will be saved in `data/alpaca_eval_model_outputs`.

3. After obtaining necessary outputs, set only `evaluate_only=True` in the config, then run `run_utility_exp_with_vllm_server.sh`. This will set up vLLM server and then do the evaluation.

4. In future improvements, steps 2 & 3 may be combined in one step. Currently, they are separated due to OOM and other technical constraints.
