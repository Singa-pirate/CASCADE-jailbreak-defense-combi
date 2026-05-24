import torch
import json

from .asr_base import AsrExperimentBase
from common.state import JailBreakGraphState
from common.label_to_class import label_to_class
from exp.build_graph import build_attack_graph, build_defense_graph
from exp.interactive_helper import greet, wait_input, stream_graph_updates
from nodes.evaluators.oss_jb_judge_20b import OssJbJudge20BNode
import logging

# NOTE for PAP experiment, must only run with "attack_only=True"
# As this implementation of attack would run evaluator and save evaluator output in final state
# After attack is done, duplicate result file and change name to reflect evaluated result; should not run evaluation again

MAX_ITERATIONS = 40 # corresponds with 40 techniques, each try at least once

# Following authors' code to obtain sampling examples
# Then have run filtering code in same folder, to filter out rejected examples using key word
sampling_example_filtered_path = "nodes/attacks/util/PAP/sampling_example_filtered.json"

# From paper: How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs
# https://github.com/CHATS-lab/persuasive_jailbreaker
class PAPExperiment(AsrExperimentBase):
    def __init__(self, **config):
        super().__init__(**config)
        optional_configs = config.get("optional_configs", {})
        self.max_iterations = optional_configs.get("max_iterations", MAX_ITERATIONS)
        # ensure only do attack for PAP experiment, reason explained in note above
        self.attack_only = True
        self.evaluate_only = False
        with open(sampling_example_filtered_path, "r") as f:
            self.sampling_example_filtered = json.load(f)

    def run_AnD_pair(self, attack_set: dict, defense_set: dict, repetitions: int) -> list[JailBreakGraphState] | None:
        attack_graph = build_attack_graph(attack_set)
        defense_graph = build_defense_graph(defense_set)
        PAP_evaluator = OssJbJudge20BNode() # to represent GPT4 judge as per paper

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
                    self.run_PAP(attack_graph, defense_graph, PAP_evaluator, init_state)
                
                logging.info(f"\n--- PAP attack {"succeeded" if is_jailbroken else "failed"} --")
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
                        self.run_PAP(attack_graph, defense_graph, PAP_evaluator, init_state)
                    states_after_AnD.append(best_state)

            del attack_graph, defense_graph, PAP_evaluator
            self.cleanup()
            return states_after_AnD
    
    def run_PAP(self, attack_graph, defense_graph, PAP_evaluator, init_state) -> tuple[bool, JailBreakGraphState, int]:
        iteration = 0
        final_state = None

        while True:
            if iteration >= self.max_iterations:
                # max number of trials reached, give up
                break
            iteration += 1
            logging.info(f"\n--- Running PAP trial {iteration} ---")

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
            
            messages = final_state["messages"]
            goal = final_state["goal"]
            prompt = final_state["prompt"]
            response = messages[-1].content
            rejected = final_state["rejected"]

            # the paper does not define evaluator to determine when to stop; so just run JB judge here, stop if jailbroken
            evaluator_state_out = {}
            jailbroken = False
            MAX_ATTEMPTS = 5 # if still fail after 5 attempts, give up and default to not jailbroken; rarely happens
            for _ in range(MAX_ATTEMPTS):
                evaluator_state_out = PAP_evaluator.evaluate_and_update_state(messages, goal, prompt, response, rejected)
                if "jailbroken" in evaluator_state_out:
                    jailbroken = evaluator_state_out["jailbroken"]
                    break

            # save evaluator state fields in final state too, no need to evaluate again
            final_state["jailbroken"] = jailbroken
            final_state["evaluator_output"] = evaluator_state_out.get("evaluator_output", "")

            if jailbroken:
                logging.info(f"\nFound successful jailbreak")
                return True, final_state, iteration

        return False, final_state, iteration
