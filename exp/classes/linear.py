import torch

from .asr_base import AsrExperimentBase
from common.state import JailBreakGraphState
from common.label_to_class import label_to_class
from exp.build_graph import build_attack_graph, build_defense_graph
from exp.interactive_helper import greet, wait_input, stream_graph_updates

class LinearExperiment(AsrExperimentBase):
    def run_AnD_pair(self, attack_set: dict, defense_set: dict, repetitions: int) -> list[JailBreakGraphState] | None:
        attack_graph = build_attack_graph(attack_set)
        defense_graph = build_defense_graph(defense_set)

        if self.interactive:
            greet()
            while True:
                user_input = wait_input()
                if user_input is None:
                    break
                
                init_state = JailBreakGraphState({
                    "goal": user_input,
                    "prompt": user_input,
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
                })
                
                state_after_attack = stream_graph_updates(attack_graph, init_state)
                stream_graph_updates(defense_graph, state_after_attack, print_final_state=False)
            return None
        else:
            dataset_label = self.evaluation_set["dataset"]
            dataset = label_to_class(dataset_label)()
            goals = dataset.load_goals()
            states_after_AnD = []

            for rep in range(repetitions):
                print(f"\n--- Repetition {rep+1}/{repetitions} ---", flush=True)
                for i, goal in enumerate(goals):
                    if i % 10 == 0:
                        print(f"Finished testing {i} goals", flush=True)
                    init_state = {
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
                    }
                    state_after_attack = attack_graph.invoke(init_state)
                    final_state = defense_graph.invoke(state_after_attack)
                    states_after_AnD.append(final_state)
            del attack_graph, defense_graph
            self.cleanup()
            return states_after_AnD
