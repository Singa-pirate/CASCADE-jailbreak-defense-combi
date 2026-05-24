from abc import ABC, abstractmethod
from common.state import JailBreakGraphState
import pandas as pd
import gc
import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from .util import save_states, load_states
from ..build_graph import build_evaluation_graph

# Default values of optional configs
EXPERIMENT_REPETITIONS = 5
OPTIMISE_MEMORY = False
EVALUATE_ONLY = False
ATTACK_ONLY = False

class AsrExperimentBase(ABC):
    def __init__(self, exp_name:str, attack_sets: list, defense_sets: list, evaluation_set: dict | None, 
                 interactive: bool, save_dir: str, optional_configs: dict):
        self.exp_name = exp_name
        self.attack_sets = attack_sets
        self.defense_sets = defense_sets
        self.evaluation_set = evaluation_set
        self.interactive = interactive
        self.save_dir = save_dir

        self.experiment_repetitions = optional_configs.get("experiment_repetitions", EXPERIMENT_REPETITIONS)
        self.optimise_memory = optional_configs.get("optimise_memory", OPTIMISE_MEMORY)
        self.evaluate_only = optional_configs.get("evaluate_only", EVALUATE_ONLY)
        self.attack_only = optional_configs.get("attack_only", ATTACK_ONLY)
        assert not (self.evaluate_only and self.attack_only), "Cannot set both evaluate_only and attack_only to True"
    
    def run(self):
        if self.interactive:
            print("\n=== Running interactive experiment ===", flush=True)
            assert len(self.attack_sets) == 1 and len(self.defense_sets) == 1
            self.run_AnD_pair(self.attack_sets[0], self.defense_sets[0], 1)
        else:
            print("\n=== Running batch experiment ===", flush=True)
            print("\n--- Attack sets ---", flush=True)
            for a_set in self.attack_sets:
                print(f"  - {a_set['name']}", flush=True)
            print("\n--- Defense sets ---", flush=True)
            for d_set in self.defense_sets:
                print(f"  - {d_set['name']}", flush=True)
            print("\n--- Evaluation set ---", flush=True)
            print(f"  - {self.evaluation_set['name']}", flush=True)
            for a_set in self.attack_sets:
                for d_set in self.defense_sets:
                    name = f"{a_set['name']}__{d_set['name']}__{self.evaluation_set['name']}"
                    llm_outputs_save_file_path = f"{self.save_dir}/{name}_LLM_outputs.parquet"
                    if self.evaluate_only:
                        # load saved LLM outputs
                        states = load_states(llm_outputs_save_file_path)
                        self.evaluate(a_set, d_set, name, states)
                    else:
                        print(f"\n--- Testing attack '{a_set['name']}' against defense '{d_set['name']}' ---", flush=True)
                        states_after_AnD = self.run_AnD_pair(a_set, d_set, self.experiment_repetitions)
                        print(f"\n--- AnD testing complete ---", flush=True)
                        # Before building evaluation graph, first save results to df
                        # In case of OOM, this saves progress
                        save_states(llm_outputs_save_file_path, states_after_AnD)
                        self.cleanup()
                        if self.attack_only:
                            continue
                        else:
                            self.evaluate(a_set, d_set, name, states_after_AnD)
                        del states_after_AnD
            print("\n=== Batch experiment complete ===", flush=True)

    @abstractmethod
    def run_AnD_pair(self, attack_set: dict, defense_set: dict, repetitions: int) -> list[JailBreakGraphState] | None:
        # NOTE: in each experiment subclass, need to implement this method for both interactive and batch mode
        # for interactive mode, return None
        # for batch mode, return list of final states for evaluation
        pass

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
        self.cleanup()
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
