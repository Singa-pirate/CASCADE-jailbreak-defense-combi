import torch

from .asr_base import AsrExperimentBase
from common.state import JailBreakGraphState
from common.label_to_class import label_to_class
from exp.build_graph import build_attack_graph, build_defense_graph
from exp.interactive_helper import greet, wait_input, stream_graph_updates
from nodes.attacks.online.renellm_attacker import ReNeLLMAttackerNode
import logging

# Default values of optional configs
MAX_ITERATIONS = 20 # NOTE for ReNeLLM, instead of "iteration", each attempt is a fresh "trial"

class ReNeLLMExperiment(AsrExperimentBase):
    def __init__(self, **config):
        super().__init__(**config)
        optional_configs = config.get("optional_configs", {})
        self.max_iterations = optional_configs.get("max_iterations", MAX_ITERATIONS)

    def run_AnD_pair(self, attack_set: dict, defense_set: dict, repetitions: int) -> list[JailBreakGraphState] | None:
        attack_graph = build_attack_graph(attack_set)
        defense_graph = build_defense_graph(defense_set)
        renellm_attacker = ReNeLLMAttackerNode()

        if self.interactive:
            greet()
            while True:
                user_input = wait_input()
                if user_input is None:
                    break
                
                goal = user_input

                init_state = {
                    "iteration": 0,
                    "target_LLM_query_count": 0,

                    "goal": goal,
                    "target": "",
                    "prompt": goal,
                    "messageHistoryForAttacker": [],
                    "messageHistoryForTargetModel": [],

                    "messages": [],
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
                }

                is_jailbroken, best_state, iteration = \
                    self.run_ReNeLLM_algo(attack_graph, defense_graph, renellm_attacker, init_state)
                
                logging.info(f"\n--- ReNeLLM attack {"succeeded" if is_jailbroken else "failed"} --")
                if is_jailbroken:
                    logging.info(f"Jailbreaking prompt:\n{best_state["prompt"]}")
                    logging.info(f"\nLLM response: {best_state["messages"][-1].content}")
                logging.info(f"Number of iterations (depth): {iteration}\n")

        else:
            dataset_label = self.evaluation_set["dataset"]
            dataset = label_to_class(dataset_label)()
            goals = dataset.load_goals()
            states_after_AnD = []

            for rep in range(repetitions):
                print(f"\n--- Repetition {rep+1}/{repetitions} ---", flush=True)

                for i in range(len(goals)):
                    if i % 10 == 0:
                        print(f"Finished testing {i} goals", flush=True)
                    goal = goals[i]
                    init_state = {
                        "iteration": 0,
                        "target_LLM_query_count": 0,

                        "goal": goal,
                        "target": "",
                        "prompt": goal,
                        "messageHistoryForAttacker": [],
                        "messageHistoryForTargetModel": [],

                        "messages": [],
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
                    }
                    is_jailbroken, best_state, iteration = \
                        self.run_ReNeLLM_algo(attack_graph, defense_graph, renellm_attacker, init_state)
                    states_after_AnD.append(best_state)

            ReNeLLMAttackerNode.cleaup_attacker_model_instance()
            del attack_graph, defense_graph, renellm_attacker
            self.cleanup()
            return states_after_AnD
    
    def run_ReNeLLM_algo(self, attack_graph, defense_graph, renellm_attacker, init_state) -> tuple[bool, JailBreakGraphState, int]:
        iteration = 0
        final_state = None

        while True:
            if iteration >= self.max_iterations:
                # max number of trials reached, give up
                break
            iteration += 1
            logging.info(f"\n--- Running ReNeLLM trial {iteration} ---")

            init_state_copy = init_state.copy()
            init_state_copy["iteration"] = iteration
            init_state_copy["target_LLM_query_count"] = iteration
            # maintain total token counts, reset other fields
            if final_state is not None:
                init_state_copy["total_input_tokens"] = final_state["total_input_tokens"]
                init_state_copy["total_output_tokens"] = final_state["total_output_tokens"]

            if self.interactive:
                final_attack_state = stream_graph_updates(attack_graph, init_state_copy)
                final_state = stream_graph_updates(defense_graph, final_attack_state)
            else:
                final_attack_state = attack_graph.invoke(init_state_copy)
                final_state = defense_graph.invoke(final_attack_state)
            
            response = final_state["messages"][-1].content
            jailbroken = renellm_attacker.harmful_classification(response)

            if jailbroken:
                logging.info(f"\nFound successful jailbreak")
                return True, final_state, iteration

        return False, final_state, iteration
