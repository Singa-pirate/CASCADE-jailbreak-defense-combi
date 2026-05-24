import json
import logging
from .base import OfflineAttackNode

# From paper: Improved Techniques for Optimization-Based Jailbreaking on Large Language Models

file_path = "nodes/attacks/util/I_GCG/I_GCG_best_adv_suffixes_vicuna.json"

class IGCGNode(OfflineAttackNode):
    def __init__(self):
        super().__init__()
        with open(file_path, 'r') as f:
            self.suffixes = json.load(f)

    def preprocess(self, prompt: str) -> str:
        prompt = prompt.strip()
        suffix = self.suffixes.get(prompt, None)
        if suffix is None:
            logging.error(f'Warning: No adversarial suffix found for prompt: {prompt}. Using original prompt.')
            return prompt
        return f'{prompt} {suffix}'
