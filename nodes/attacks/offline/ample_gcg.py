import json
import logging

from .base import OfflineAttackNode

# From paper: AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs

n_suffixes = 5
with open("nodes/attacks/util/ampleGCG_adv_suffixes.json", "r") as f:
    adv_suffixes_dict = json.load(f)

class AmpleGCGNode(OfflineAttackNode):
    def __init__(self, index:int=0):
        super().__init__()
        if index < 0 or index >= n_suffixes:
            raise ValueError(f'Index must be between 0 and {n_suffixes-1}')
        self.index = index

    def preprocess(self, prompt: str) -> str:
        prompt = prompt.strip()
        if prompt in adv_suffixes_dict:
            adv_suffix = adv_suffixes_dict[prompt][self.index]
            return f"{prompt} {adv_suffix}"
        else:
            logging.error("Warning: Prompt not found in AmpleGCG dict, using original prompt")
            return prompt
