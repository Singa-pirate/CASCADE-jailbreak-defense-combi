import copy
import gc
import json
import logging
import os
from csv import DictWriter
from typing import Any

import torch
from alpaca_eval.main import evaluate

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover - optional in some local envs
    dist = None

try:
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
except Exception:  # pragma: no cover - optional in utility-only envs
    def cleanup_dist_env_and_memory():
        return None

from exp.build_graph import build_defense_graph
from common.constants import REJECTION_MESSAGE
from common.helpers.batch_helper import (
    OpenAIBatchHelper,
    GCPClaudeBatchHelper,
    AnthropicClaudeBatchHelper,
    GeminiBatchHelper,
)
from .util import save_states, load_states


# Default values of optional configs
EXPERIMENT_REPETITIONS = 1
REFERENCE_OUTPUTS_FILENAME = "reference_model.json"
ALPACA_EVAL_MODEL_OUTPUTS_PATH = "data/alpaca_eval_model_outputs"

RUN_DEFENSE_BEFORE_LLM_AND_GENERATE_BATCH = True
SUBMIT_BATCH_FILE = False
RUN_DEFENSE_AFTER_LLM = False
RUN_EVALUATION = False

MODEL = "gpt-3.5-turbo-0125"
MODEL_KWARGS = {}
CLAUDE_BATCH_PROVIDER = "anthropic"  # options: gcp, anthropic


class BatchAPIUtilityExperiment:
    """Generate AlpacaEval utility outputs through a batch API target model.

    Workflow:
    1. Run defense_sequence_before_llm on AlpacaEval instructions, save states, create batch JSONL.
    2. After the batch output is downloaded to <name>_batch_output.jsonl, merge target LLM
       outputs into the saved states, run defense_sequence_after_llm, then save AlpacaEval
       model outputs JSON for later utility evaluation.
    3. Optionally run AlpacaEval on the generated JSON.

    This class intentionally mirrors BatchAPIExperiment, but replaces attack generation with
    AlpacaEval utility instructions.
    """

    def __init__(self, exp_name: str, defense_sets: list, save_dir: str, optional_configs: dict):
        self.exp_name = exp_name
        self.defense_sets = defense_sets
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.experiment_repetitions = optional_configs.get(
            "experiment_repetitions", EXPERIMENT_REPETITIONS
        )
        if self.experiment_repetitions != 1:
            logging.warning(
                "UtilityExperiment currently fixes experiment_repetitions to 1; "
                "batch utility generation will keep the configured value."
            )

        self.run_defense_before_llm_and_generate_batch = optional_configs.get(
            "run_defense_before_llm_and_generate_batch",
            RUN_DEFENSE_BEFORE_LLM_AND_GENERATE_BATCH,
        )
        self.submit_batch_file = optional_configs.get("submit_batch_file", SUBMIT_BATCH_FILE)
        self.run_defense_after_llm = optional_configs.get(
            "run_defense_after_llm", RUN_DEFENSE_AFTER_LLM
        )
        self.run_evaluation = optional_configs.get("run_evaluation", RUN_EVALUATION)

        self.model = optional_configs.get("model", MODEL)
        self.model_kwargs = optional_configs.get("model_kwargs", MODEL_KWARGS)
        self.claude_batch_provider = optional_configs.get(
            "claude_batch_provider", CLAUDE_BATCH_PROVIDER
        )
        self.batch_helper = self._build_batch_helper()

        self.reference_outputs_filename = optional_configs.get(
            "reference_outputs_filename", REFERENCE_OUTPUTS_FILENAME
        )
        self.alpaca_eval_model_outputs_path = optional_configs.get(
            "alpaca_eval_model_outputs_path", ALPACA_EVAL_MODEL_OUTPUTS_PATH
        )
        self.reference_outputs_path = os.path.join(
            self.alpaca_eval_model_outputs_path, self.reference_outputs_filename
        )
        if not os.path.exists(self.reference_outputs_path):
            raise ValueError(
                f"Reference outputs file {self.reference_outputs_path} does not exist. "
                "Please check your config."
            )
        with open(self.reference_outputs_path, "r") as f:
            self.eval_set = json.load(f)

        self.subfolder_name = optional_configs.get("subfolder_name", None)
        self.output_subfolder = optional_configs.get("output_subfolder", self.subfolder_name)
        self.annotators_config = optional_configs.get(
            "annotators_config", "weighted_alpaca_eval_vllm_llama3_70b"
        )

        # Backward-compatible alias from the sequential utility experiment.
        generate_and_evaluate = optional_configs.get("generate_and_evaluate", None)
        if generate_and_evaluate is True:
            self.run_evaluation = True
        elif generate_and_evaluate is False:
            self.run_defense_before_llm_and_generate_batch = True

        assert (
            self.run_defense_before_llm_and_generate_batch
            or self.run_defense_after_llm
            or self.run_evaluation
        ), (
            "In batch utility exp, at least one of "
            "run_defense_before_llm_and_generate_batch, run_defense_after_llm, "
            "run_evaluation must be True"
        )
        assert not (
            self.run_defense_before_llm_and_generate_batch
            and (self.run_defense_after_llm or self.run_evaluation)
        ), (
            "In batch utility exp, if run_defense_before_llm_and_generate_batch is True, "
            "run_defense_after_llm and run_evaluation must be False. Run later steps "
            "after the batch output file has been downloaded."
        )

    def _build_batch_helper(self):
        if self.model.startswith("gpt"):
            return OpenAIBatchHelper()
        elif self.model.startswith("claude"):
            if self.claude_batch_provider == "gcp":
                return GCPClaudeBatchHelper()
            if self.claude_batch_provider == "anthropic":
                return AnthropicClaudeBatchHelper()
            raise ValueError(
                f"Unrecognized claude_batch_provider '{self.claude_batch_provider}'. "
                "Supported values are 'gcp' and 'anthropic'."
            )
        elif self.model.startswith("gemini"):
            return GeminiBatchHelper()
        else:
            raise ValueError(
                f"Unrecognized model {self.model}. Currently only supports 'gpt*','claude*' and 'gemini*' models."
            )

    def run(self):
        print("\n=== Running batch utility evaluation with AlpacaEval ===", flush=True)
        print("\n--- Defense sets ---", flush=True)
        for d_set in self.defense_sets:
            print(f"  - {d_set['name']}", flush=True)
        print("\n--- Batch API model ---", flush=True)
        print(f"  - {self.model} with kwargs {self.model_kwargs}", flush=True)
        print("\n--- Reference outputs ---", flush=True)
        print(f"  - {self.reference_outputs_path}", flush=True)

        for d_set in self.defense_sets:
            name = d_set["name"]
            before_target_llm_save_path = os.path.join(
                self.save_dir, f"{name}_before_target_LLM.parquet"
            )
            batch_jsonl_path = os.path.join(self.save_dir, f"{name}_batch.jsonl")
            batch_output_path = os.path.join(self.save_dir, f"{name}_batch_output.jsonl")
            model_output_save_path = self._model_output_path(name)

            if self.run_defense_before_llm_and_generate_batch:
                states, batch_line_list = self.generate_batch(d_set, name)
                save_states(before_target_llm_save_path, states)

                if len(batch_line_list) == 0:
                    logging.error(
                        f"Warning: for '{name}', after defenses before LLM, no prompt is left "
                        "to query. No batch file is created or submitted."
                    )
                else:
                    self.batch_helper.create_batch_jsonl(batch_jsonl_path, batch_line_list)
                    ret = self.batch_helper.upload_and_submit_batch(
                        jsonl_path=batch_jsonl_path,
                        description=f"utility__{name}",
                        model=self.model,
                        submit=self.submit_batch_file,
                    )
                    if self.submit_batch_file and ret is not None:
                        self._record_batch_submission(name, ret)
                self.cleanup()

            if self.run_defense_after_llm:
                states_before_target_llm = load_states(before_target_llm_save_path)
                batch_output = self.batch_helper.parse_batch_output(batch_output_path)
                final_examples = self.defense_after_llm(
                    d_set=d_set,
                    name=name,
                    states_before_target_llm=states_before_target_llm,
                    batch_output=batch_output,
                )
                os.makedirs(os.path.dirname(model_output_save_path), exist_ok=True)
                with open(model_output_save_path, "w") as f:
                    json.dump(final_examples, f, indent=2)
                print(f"\n--- Utility outputs saved to {model_output_save_path} ---", flush=True)
                self.cleanup()

            if self.run_evaluation:
                self.evaluate(d_set, name, model_output_save_path)
                self.cleanup()

        print("\n=== Batch utility experiment complete ===", flush=True)

    def _model_output_path(self, name: str) -> str:
        output_dir = self.alpaca_eval_model_outputs_path
        if self.output_subfolder is not None:
            output_dir = os.path.join(output_dir, self.output_subfolder)
        return os.path.join(output_dir, f"{name}.json")

    def _record_batch_submission(self, name: str, ret: Any):
        print(f"\nBatch file for '{name}' uploaded successfully.", flush=True)
        batch_submission_csv_path = os.path.join(self.save_dir, "batch_submission.csv")
        submission_field = getattr(self.batch_helper, "submission_csv_field", "batch_id")
        with open(batch_submission_csv_path, "a") as f:
            writer = DictWriter(f, fieldnames=[submission_field, "name", "kind"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({submission_field: ret, "name": name, "kind": "utility"})

    def generate_batch(self, d_set: dict, name: str) -> tuple[list[dict], list[dict]]:
        print(f"\n--- Running defenses before LLM and generating batch file for '{name}' ---", flush=True)

        d_set_before_llm = copy.deepcopy(d_set)
        if d_set_before_llm.get("target_llm") != "LLM.PLACEHOLDER":
            logging.error(
                "Warning: In batch utility exp, target LLM must be LLM.PLACEHOLDER "
                "during batch generation, overriding."
            )
            d_set_before_llm["target_llm"] = "LLM.PLACEHOLDER"
        d_set_before_llm["defense_sequence_after_llm"] = []

        defense_graph_before_llm = build_defense_graph(
            d_set_before_llm, add_rejection_helper=False
        )

        batch_line_list = []
        states = []
        eval_set_copy = copy.deepcopy(self.eval_set)
        for rep in range(self.experiment_repetitions):
            for i, example in enumerate(eval_set_copy):
                if i % 10 == 0:
                    print(
                        f" - Finished preparing {i} / {len(eval_set_copy)} examples "
                        f"for repetition {rep}",
                        flush=True,
                    )

                instruction = example["instruction"]
                init_state = {
                    "goal": instruction,
                    "prompt": instruction,
                    "messages": [],
                    "messageHistoryForTargetModel": [],
                    "rejected": False,
                    "rejector": "",
                    "reject_reason": "",
                    "jailbroken": False,
                    "evaluator_output": "",
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "max_output_tokens_reached": False,
                    "example_id": i,
                    "goal_id": i,
                    "repetition": rep,
                    "alpaca_eval_example": copy.deepcopy(example),
                }
                final_state = defense_graph_before_llm.invoke(init_state)
                states.append(final_state)

                if final_state.get("rejected", False):
                    continue
                prompt = final_state.get("prompt", "")
                if prompt is None or str(prompt).strip() == "":
                    continue

                custom_id = self._custom_id(i, rep)
                messages = [{"role": "user", "content": str(prompt)}]
                batch_line = self.batch_helper.create_batch_line(
                    custom_id,
                    self.model,
                    messages,
                    **self.model_kwargs,
                )
                batch_line_list.append(batch_line)

        del defense_graph_before_llm, eval_set_copy
        return states, batch_line_list

    def defense_after_llm(
        self,
        d_set: dict,
        name: str,
        states_before_target_llm: list[dict],
        batch_output: list[dict],
    ) -> list[dict]:
        print(f"\n--- Running defenses after LLM for '{name}' ---", flush=True)

        batch_output_dict = self._batch_output_by_custom_id(batch_output)

        d_set_after_llm = copy.deepcopy(d_set)
        d_set_after_llm["defense_sequence_before_llm"] = []
        d_set_after_llm["target_llm"] = None
        defense_graph_after_llm = build_defense_graph(
            d_set_after_llm, add_rejection_helper=True
        )

        # Keep AlpacaEval ordering deterministic by index, not by batch output ordering.
        examples_by_id = {
            i: copy.deepcopy(example) for i, example in enumerate(copy.deepcopy(self.eval_set))
        }

        for state in states_before_target_llm:
            example_id = state.get("example_id", state.get("goal_id", None))
            repetition = state.get("repetition", 0)
            if repetition != 0:
                logging.warning(
                    "AlpacaEval expects one output per instruction; writing repetition 0 only."
                )
                continue
            if example_id is None:
                logging.error(f"Warning: skipping state without example_id: {state}")
                continue

            example = examples_by_id[int(example_id)]
            example["generator"] = name

            if state.get("rejected", False):
                example["output"] = REJECTION_MESSAGE
                continue

            custom_id = self._custom_id(example_id, repetition)
            if custom_id not in batch_output_dict:
                logging.error(
                    f"Warning: batch output missing for {custom_id}; writing empty output."
                )
                example["output"] = ""
                continue

            response_body = batch_output_dict[custom_id]
            llm_response = self._extract_text_from_response_body(response_body)
            input_tokens, output_tokens = self._extract_usage(response_body)

            state["messages"].append({"role": "assistant", "content": llm_response})
            state["response"] = llm_response
            state["total_input_tokens"] = input_tokens
            state["total_output_tokens"] = output_tokens

            state_after_defense = defense_graph_after_llm.invoke(state)
            if state_after_defense.get("rejected", False):
                example["output"] = REJECTION_MESSAGE
            else:
                example["output"] = self._extract_output_from_state(state_after_defense)

        del defense_graph_after_llm
        return [examples_by_id[i] for i in sorted(examples_by_id.keys())]

    def _batch_output_by_custom_id(self, batch_output: list[dict]) -> dict[str, dict]:
        batch_output_dict = {}
        for line in batch_output:
            custom_id = line.get("custom_id")
            response = line.get("response", {})
            status_code = response.get("status_code", 200)
            if status_code != 200:
                logging.error(f"Warning: skipping line with non-200 status code: {line}")
                continue
            body = response.get("body", line.get("result", line))
            batch_output_dict[custom_id] = body
        return batch_output_dict

    def _extract_text_from_response_body(self, body: dict) -> str:
        """Extract assistant text from OpenAI, Anthropic, or GCP Claude batch bodies."""
        if body is None:
            return ""

        # OpenAI Chat Completions / compatible response
        choices = body.get("choices") if isinstance(body, dict) else None
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )

        # Anthropic Messages API response
        content = body.get("content") if isinstance(body, dict) else None
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict):
                    chunks.append(part.get("text", ""))
                else:
                    chunks.append(str(part))
            return "".join(chunks)

        # Some GCP Claude wrappers nest prediction payloads.
        predictions = body.get("predictions") if isinstance(body, dict) else None
        if predictions:
            first_prediction = predictions[0]
            if isinstance(first_prediction, dict):
                return self._extract_text_from_response_body(first_prediction)
            return str(first_prediction)

        return str(body)

    def _extract_usage(self, body: dict) -> tuple[int, int]:
        if not isinstance(body, dict):
            return 0, 0
        usage = body.get("usage", {}) or {}
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("inputTokenCount")
            or 0
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("outputTokenCount")
            or 0
        )
        return int(input_tokens), int(output_tokens)

    def _extract_output_from_state(self, state: dict) -> str:
        if state.get("response") is not None:
            return str(state.get("response"))
        messages = state.get("messages", [])
        if len(messages) == 0:
            return ""
        last_message = messages[-1]
        if isinstance(last_message, dict):
            return str(last_message.get("content", ""))
        return str(getattr(last_message, "content", last_message))

    def _custom_id(self, example_id: int, repetition: int = 0) -> str:
        return f"example{example_id}__rep{repetition}"

    def evaluate(self, d_set: dict, name: str, model_output_save_path: str):
        print(f"\n--- Evaluating generated utility outputs for '{name}' ---", flush=True)
        result_output_path = os.path.join(self.save_dir, f"{name}.txt")
        if not os.path.exists(model_output_save_path):
            print(f"  - {model_output_save_path} does not exist, skipping evaluation", flush=True)
            return

        leaderboard, _ = evaluate(
            model_outputs=model_output_save_path,
            reference_outputs=self.reference_outputs_path,
            annotators_config=self.annotators_config,
            is_return_instead_of_print=True,
        )
        with open(result_output_path, "w") as f:
            f.write("--- Defense config ---\n")
            for d in d_set["defense_sequence_before_llm"]:
                f.write(f"{d}\n")
            f.write(f"{d_set['target_llm']}\n")
            for d in d_set["defense_sequence_after_llm"]:
                f.write(f"{d}\n")

            result = leaderboard[
                ["length_controlled_winrate", "lc_standard_error", "win_rate", "standard_error"]
            ]
            f.write("\n--- Evaluation results ---\n")
            f.write(result.to_string())

        print(f"\n--- Utility evaluation saved to {result_output_path} ---", flush=True)

    def cleanup(self):
        if dist is not None and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        cleanup_dist_env_and_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        gc.collect()
