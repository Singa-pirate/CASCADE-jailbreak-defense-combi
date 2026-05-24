import json
import logging

from .base import OnlineAttackNode

# Following authors' code to obtain sampling examples
# Then have run filtering code in same folder, to filter out rejected examples using key word
sampling_example_filtered_path = "nodes/attacks/util/PAP/sampling_example_filtered.json"

# From paper: How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs
# https://github.com/CHATS-lab/persuasive_jailbreaker
class PAPAttackerNode(OnlineAttackNode):
    def __init__(self):
        super().__init__()
        with open(sampling_example_filtered_path, "r") as f:
            self.sampling_example_filtered = json.load(f)

    def improve(self, iteration: int, goal: str, target: str, prompt: str, messages: list, message_history_for_attacker:list, reject_reason: str) -> str:
        if prompt not in self.sampling_example_filtered:
            logging.error("Warning: prompt not found in sampling example file, returning original prompt")
            return prompt
        sampling_example_list = self.sampling_example_filtered[prompt]
        # If there's no attack example sampled, return original prompt
        if len(sampling_example_list) == 0:
            if iteration == 0:
                # notify for the first time
                logging.error("Warning: for this prompt, attacker rejected to sample any attack prompt, will try with original prompt")
            return prompt
        else:
            # try each sampling example in a rotating way
            return list(sampling_example_list[iteration % len(sampling_example_list)].values())[0]
