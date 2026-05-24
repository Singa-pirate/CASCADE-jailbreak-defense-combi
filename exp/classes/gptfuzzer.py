from exp.classes.asr_base import AsrExperimentBase
from common.state import JailBreakGraphState
from common.label_to_class import label_to_class
from exp.build_graph import build_attack_graph, build_defense_graph
from exp.interactive_helper import greet, wait_input, stream_graph_updates
import logging

from nodes.attacks.online.gptfuzzer_attacker import GPTFuzzerAttackerNode
from nodes.attacks.util.GPTFuzzer.mutator import Mutator, MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand, OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten
from nodes.attacks.util.GPTFuzzer.selection import MCTSExploreSelectPolicy
from nodes.evaluators.gptfuzzer_evaluator import GPTFuzzerEvaluator
from nodes.llms.oss_20b import Oss20B

class GPTFuzzerExperiment(AsrExperimentBase):
    def __init__(self, **config):
        super().__init__(**config)
        optional_configs = config.get("optional_configs", {})

    def run_AnD_pair(self, attack_set: dict, defense_set: dict, repetitions: int) -> list[JailBreakGraphState] | None:
        attack_graph = build_attack_graph(attack_set)
        defense_graph = build_defense_graph(defense_set)
        attacker_model = Oss20B()
        GPTFuzzer_evaluator = GPTFuzzerEvaluator()
        GPTFuzzer_attacker = GPTFuzzerAttackerNode.init_fuzzer_instance(
            mutate_policy=MutateRandomSinglePolicy([
                OpenAIMutatorCrossOver(attacker_model),
                OpenAIMutatorExpand(attacker_model),
                OpenAIMutatorGenerateSimilar(attacker_model),
                OpenAIMutatorRephrase(attacker_model),
                OpenAIMutatorShorten(attacker_model)],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy()
        )

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

                is_jailbroken, best_state, iteration, query_count = \
                    self.run_GPTFuzzer_algo(attack_graph, defense_graph, GPTFuzzer_attacker, GPTFuzzer_evaluator, init_state)
                
                logging.info(f"\n--- GPTFuzzer attack {"succeeded" if is_jailbroken else "failed"} --")
                if is_jailbroken:
                    logging.info(f"Jailbreaking prompt:\n{best_state["prompt"]}")
                    logging.info(f"\nLLM response: {best_state["messages"][-1].content}")
                logging.info(f"Number of iterations (depth): {iteration}\nNumber of queries: {query_count}")

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
                    is_jailbroken, best_state, iteration, query_count = \
                        self.run_GPTFuzzer_algo(attack_graph, defense_graph, GPTFuzzer_attacker, GPTFuzzer_evaluator, init_state)
                    states_after_AnD.append(best_state)

            del attack_graph, defense_graph, attacker_model, GPTFuzzer_attacker, GPTFuzzer_evaluator
            GPTFuzzerAttackerNode.cleaup_fuzzer_instance()
            self.cleanup()
            return states_after_AnD
    
    def run_GPTFuzzer_algo(self, attack_graph, defense_graph, GPTFuzzer_attacker, GPTFuzzer_evaluator, init_state) -> tuple[bool, JailBreakGraphState, int, int]:
        iteration = 0
        final_state = None
        total_input_tokens = 0
        total_output_tokens = 0

        while True:
            iteration += 1
            logging.info(f"\n--- Running GPTFuzzer trial {iteration} ---")

            init_state_copy = init_state.copy()

            if self.interactive:
                final_attack_state = stream_graph_updates(attack_graph, init_state_copy)
                final_state = stream_graph_updates(defense_graph, final_attack_state)
            else:
                final_attack_state = attack_graph.invoke(init_state_copy)
                final_state = defense_graph.invoke(final_attack_state)
            
            response = final_state["messages"][-1].content
            results = GPTFuzzer_evaluator.predict([response])
            jailbroken = results[0] == 1

            total_input_tokens += final_state["total_input_tokens"]
            total_output_tokens += final_state["total_output_tokens"]

            if jailbroken:
                logging.info(f"\nFound successful jailbreak")
                GPTFuzzerAttackerNode.update(jailbroken=True)
            else:
                GPTFuzzerAttackerNode.update(jailbroken=False)
            
            if GPTFuzzer_attacker.is_stop():
                final_state["iteration"] = iteration
                query_count = GPTFuzzer_attacker.current_query
                final_state["target_LLM_query_count"] = query_count
                final_state["total_input_tokens"] = total_input_tokens
                final_state["total_output_tokens"] = total_output_tokens
                GPTFuzzerAttackerNode.reset()
                return jailbroken, final_state, iteration, query_count
