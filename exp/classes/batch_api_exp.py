from common.state import JailBreakGraphState
import pandas as pd
import gc
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from csv import DictWriter
import logging

from .util import save_states, load_states
from ..build_graph import build_attack_graph, build_defense_graph, build_evaluation_graph
from common.label_to_class import label_to_class
from common.helpers.batch_helper import OpenAIBatchHelper, GCPClaudeBatchHelper, AnthropicClaudeBatchHelper

# Default values of optional configs
EXPERIMENT_REPETITIONS = 5
RUN_ATTACK_AND_GENERATE_BATCH = True # from dataset, run attack graph & defense before LLM, save as pq, then create batch file
SUBMIT_BATCH_FILE = False # submit batch file; if False, will upload but not submit, need to manually submit in dashboard
RUN_DEFENSE_AFTER_LLM = False # load batch result & pq, run defense graph after LLM output, save as final pq
RUN_EVALUATION = False # load final pq, run evaluation, save results
MODEL = "gpt-3.5-turbo-0125"
MODEL_KWARGS = {}
CLAUDE_BATCH_PROVIDER = "anthropic" # options: gcp, anthropic

# This class is for batch API experiments
# where the complete experiment is split into 3 (async) steps:
# 1) run attack graph until target LLM, save states as pq, generate batch file and submit
# 2) download batch output, update states with LLM output and run defense graph after LLM, save final states as pq
# 3) run evaluation graph on final states and save results
class BatchAPIExperiment():
    def __init__(self, exp_name:str, attack_sets: list, defense_sets: list, evaluation_set: dict | None, 
                 save_dir: str, optional_configs: dict):
        self.exp_name = exp_name
        self.attack_sets = attack_sets
        self.defense_sets = defense_sets
        self.evaluation_set = evaluation_set
        self.save_dir = save_dir

        self.experiment_repetitions = optional_configs.get("experiment_repetitions", EXPERIMENT_REPETITIONS)
        self.run_attack_and_generate_batch = optional_configs.get("run_attack_and_generate_batch", RUN_ATTACK_AND_GENERATE_BATCH)
        self.submit_batch_file = optional_configs.get("submit_batch_file", SUBMIT_BATCH_FILE)
        self.run_defense_after_llm = optional_configs.get("run_defense_after_llm", RUN_DEFENSE_AFTER_LLM)
        self.run_evaluation = optional_configs.get("run_evaluation", RUN_EVALUATION)
        self.model = optional_configs.get("model", MODEL)
        self.model_kwargs = optional_configs.get("model_kwargs", MODEL_KWARGS)
        self.claude_batch_provider = optional_configs.get("claude_batch_provider", CLAUDE_BATCH_PROVIDER)
        if self.model.startswith("gpt"):
            self.batch_helper = OpenAIBatchHelper()
            if self.model.startswith("gpt-5") and self.model_kwargs.get("reasoning_effort", None) is None:
                # default to none is not set
                self.model_kwargs["reasoning_effort"] = "none"
        elif self.model.startswith("claude"):
            if self.claude_batch_provider == "gcp":
                self.batch_helper = GCPClaudeBatchHelper()
            elif self.claude_batch_provider == "anthropic":
                self.batch_helper = AnthropicClaudeBatchHelper()
            else:
                raise ValueError(
                    f"Unrecognized claude_batch_provider '{self.claude_batch_provider}'. "
                    "Supported values are 'gcp' and 'anthropic'."
                )
        else:
            raise ValueError(f"Unrecognized model {self.model}. Currently only supports 'gpt*' and 'claude*' models.")
        assert (self.run_attack_and_generate_batch or self.run_defense_after_llm or self.run_evaluation), \
            "In batch API exp, at least one of run_attack_and_generate_batch, run_defense_after_llm, run_evaluation must be True"
        assert not (self.run_attack_and_generate_batch and (self.run_defense_after_llm or self.run_evaluation)), \
            "In batch API exp, if run_attack_and_generate_batch is True, run_defense_after_llm & run_evaluation must be both False, and vice versa"
    
    def run(self):
        print("\n=== Running batch experiment ===", flush=True)

        print("\n--- Attack sets ---", flush=True)
        for a_set in self.attack_sets:
            print(f"  - {a_set['name']}", flush=True)
        print("\n--- Defense sets ---", flush=True)
        for d_set in self.defense_sets:
            print(f"  - {d_set['name']}", flush=True)
        print("\n--- Batch API model ---", flush=True)
        print(f"  - {self.model} with kwargs {self.model_kwargs}", flush=True)
        print("\n--- Evaluation set ---", flush=True)
        print(f"  - {self.evaluation_set['name']}", flush=True)

        for a_set in self.attack_sets:
            for d_set in self.defense_sets:
                name = f"{a_set['name']}__{d_set['name']}__{self.evaluation_set['name']}"
                before_target_llm_save_path = f"{self.save_dir}/{name}_before_target_LLM.parquet"
                batch_jsonl_path = f"{self.save_dir}/{name}_batch.jsonl"
                batch_output_path = f"{self.save_dir}/{name}_batch_output.jsonl" # NOTE: expected to be renamed by user after downloading batch output file
                llm_output_save_path = f"{self.save_dir}/{name}_LLM_outputs.parquet"

                # Step 1: run attack and generate batch file
                # This will run until target LLM in defense graph
                if self.run_attack_and_generate_batch:
                    final_states, batch_line_list = self.attack_and_generate_batch(a_set, d_set, name)
                    save_states(before_target_llm_save_path, final_states)
                    if len(batch_line_list) == 0:
                        logging.error(f"Warning: for '{name}', after defenses before LLM, no prompt is left to query.\nNo batch file is created or submitted.")
                    else:
                        self.batch_helper.create_batch_jsonl(batch_jsonl_path, batch_line_list)
                        ret = self.batch_helper.upload_and_submit_batch(
                            jsonl_path=batch_jsonl_path,
                            description=name,
                            model=self.model,
                            submit=self.submit_batch_file
                        )
                        if self.submit_batch_file and ret is not None:
                            print(f"\nBatch file for '{name}' uploaded successfully.", flush=True)
                            # save batch info for downloading later
                            batch_submission_csv_path = f"{self.save_dir}/batch_submission.csv"
                            submission_field = getattr(self.batch_helper, "submission_csv_field", "batch_id")
                            with open(batch_submission_csv_path, 'a') as f:
                                writer = DictWriter(f, fieldnames=[submission_field, 'name'])
                                if f.tell() == 0:
                                    writer.writeheader()
                                writer.writerow({submission_field: ret, 'name': name})
                    self.cleanup()
                
                # Step 2: update states with batch output & run defense after LLM
                if self.run_defense_after_llm:
                    states_before_target_llm = load_states(before_target_llm_save_path)
                    batch_output = self.batch_helper.parse_batch_output(batch_output_path)
                    final_states_after_defense = self.defense_after_llm(d_set, name, states_before_target_llm, batch_output)
                    save_states(llm_output_save_path, final_states_after_defense)
                    self.cleanup()

                # Step 3: evaluate
                if self.run_evaluation:
                    states_to_evaluate = load_states(llm_output_save_path)
                    self.evaluate(a_set, d_set, name, states_to_evaluate)
                    self.cleanup()

        print("\n=== Batch experiment complete ===", flush=True)


    def attack_and_generate_batch(self, a_set: dict, d_set: dict, name: str) -> tuple[list[JailBreakGraphState], list[dict]]:
        print(f"\n--- Running attack and generating batch file for '{name}' ---", flush=True)
        d_set_copy = d_set.copy()
        if d_set_copy["target_llm"] != "LLM.PLACEHOLDER":
            logging.error("Warning: In batch API exp, target LLM must be LLM.PLACEHOLDER, overriding.")
            d_set_copy["target_llm"] = "LLM.PLACEHOLDER"
        d_set_copy["defense_sequence_after_llm"] = []
        a_graph = build_attack_graph(a_set)
        d_graph = build_defense_graph(d_set_copy, add_rejection_helper=False)
        dataset_label = self.evaluation_set["dataset"]
        dataset = label_to_class(dataset_label)()

        batch_line_list = []
        final_states = []
        for rep in range(self.experiment_repetitions):
            for i, goal in enumerate(dataset.load_goals()):
                init_state = JailBreakGraphState({
                    "goal": goal,
                    "prompt": goal,
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
                    "goal_id": i,
                    "repetition": rep,
                })
                state_after_attack = a_graph.invoke(init_state)
                final_state = d_graph.invoke(state_after_attack)
                final_states.append(final_state)
                if final_state.get("rejected", False):
                    continue # skip if rejected before target LLM
                if final_state.get("prompt", "").strip() == "":
                    continue # skip if prompt is empty for some reason
                custom_id = f"goal{i}__rep{rep}"
                messages = [{
                    "role": "user",
                    "content": final_state.get("prompt")
                }]
                batch_line = self.batch_helper.create_batch_line(custom_id, self.model, messages, **self.model_kwargs)
                batch_line_list.append(batch_line)
        del a_graph, d_graph, dataset
        return final_states, batch_line_list


    def defense_after_llm(self, d_set: dict, name: str, states_before_target_llm: list[JailBreakGraphState], \
                          batch_output: list[dict]) -> list[JailBreakGraphState]:
        print(f"\n--- Running defense after LLM for '{name}' ---", flush=True)
        batch_output_dict = {}
        for line in batch_output:
            if line["response"]["status_code"] != 200:
                print(f"Warning: skipping line with non-200 status code: {line}", flush=True)
            else:
                batch_output_dict[line["custom_id"]] = line["response"]["body"]
        for state in states_before_target_llm:
            goal_id = state.get("goal_id", None)
            repetition = state.get("repetition", None)
            custom_id = f"goal{goal_id}__rep{repetition}"
            if custom_id not in batch_output_dict:
                continue # skipped in previous step
            llm_response = batch_output_dict[custom_id]["choices"][0]["message"]["content"]
            target_llm_input_tokens = batch_output_dict[custom_id]["usage"]["prompt_tokens"]
            target_llm_output_tokens = batch_output_dict[custom_id]["usage"]["completion_tokens"]

            # update state with target LLM output
            state["messages"].append({"role": "assistant", "content": llm_response})
            state["response"] = llm_response
            state["total_input_tokens"] = target_llm_input_tokens
            state["total_output_tokens"] = target_llm_output_tokens

        d_set_copy = d_set.copy()
        d_set_copy["defense_sequence_before_llm"] = []
        d_set_copy["target_llm"] = None
        d_graph_after_llm = build_defense_graph(d_set_copy, add_rejection_helper=True)
        final_states_after_defense = []
        for state in states_before_target_llm:
            state_after_defense = d_graph_after_llm.invoke(state)
            final_states_after_defense.append(state_after_defense)
        del d_graph_after_llm
        return final_states_after_defense


    def evaluate(self, a_set: dict, d_set: dict, name: str, states: list[JailBreakGraphState]):
        # Build evaluation graph
        e_graph = build_evaluation_graph(self.evaluation_set)
        success_count = 0
        
        print(f"\n--- Evaluating with '{self.evaluation_set['name']}' ---", flush=True)
        evaluated_states = []
        for i, state in enumerate(states):
            if i % 10 == 0:
                print(f"Finished evaluating {i} conversations", flush=True)
            
            evaluated = e_graph.invoke(state)
            evaluated_states.append(evaluated)
            if evaluated.get("jailbroken", False):
                success_count += 1

        # save results
        save_states(f"{self.save_dir}/{name}.parquet", evaluated_states)

        # calculate ASR metrics
        average_asr = success_count / len(evaluated_states)
        df = pd.DataFrame(evaluated_states)
        max_asr = df.groupby('goal')['jailbroken'].max().mean()

        with open(f"{self.save_dir}/{name}.txt", 'w') as f:
            f.write("--- Attack config ---\n")
            for a in a_set["attack_sequence"]:
                f.write(f"{a}\n")
            
            f.write("\n--- Defense config ---\n")
            for d in d_set["defense_sequence_before_llm"]:
                f.write(f"{d}\n")
            f.write(f"{d_set['target_llm']}\n")
            for d in d_set["defense_sequence_after_llm"]:
                f.write(f"{d}\n")
            
            f.write("\n--- Evaluation config ---\n")
            f.write(f"{self.evaluation_set['dataset']}\n")
            for e in self.evaluation_set["evaluator_sequence"]:
                f.write(f"{e}\n")

            f.write(f"\n--- Results ---\n")
            f.write(f"average_ASR={average_asr}\n")
            f.write(f"max_ASR={max_asr}\n")
        
        del e_graph, evaluated_states, df
        print(f"\n--- Evaluation complete. Results saved to {self.save_dir} ---", flush=True)
    
    def cleanup(self):
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        cleanup_dist_env_and_memory()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        gc.collect()
