import logging
import gc
import torch
from typing import TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from nodes.attacks.util.GPTFuzzer.mutator import Mutator
    from nodes.attacks.util.GPTFuzzer.selection import MCTSExploreSelectPolicy

from .base import OnlineAttackNode
from nodes.attacks.util.GPTFuzzer.template import synthesis_message

## From paper: GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts
# https://github.com/sherdencooper/GPTFuzz
# NOTE: tweaked implementation of energy: assume always 1 energy, only mutate 1 prompt each time; so that LLM implementation can stay the same
class PromptNode:
    def __init__(self,
                 fuzzer: 'GPTFuzzer',
                 prompt: str,
                 response: str = None,
                 results: 'list[int]' = None,
                 parent: 'PromptNode' = None,
                 mutator: 'Mutator' = None):
        self.fuzzer: 'GPTFuzzer' = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: 'list[int]' = results
        self.visited_num = 0

        self.parent: 'PromptNode' = parent
        self.mutator: 'Mutator' = mutator
        self.child: 'list[PromptNode]' = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        return sum(self.results)

    @property
    def num_reject(self):
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        return len(self.results)


class GPTFuzzer:
    _attacker_model = None

    def __init__(self, mutate_policy, select_policy, max_query=100, max_jailbreak=1, concatenate=True):
        super().__init__()
        self.initial_seed = pd.read_csv("nodes/attacks/util/GPTFuzzer/GPTFuzzer.csv")["text"].tolist()
        self.mutate_policy = mutate_policy
        self.select_policy = select_policy
        self.max_query = max_query
        self.max_jailbreak = max_jailbreak
        # self.max_reject: int = -1
        # self.max_iteration: int = -1
        self.energy = 1
        
        self.prompt_nodes: 'list[PromptNode]' = [
            PromptNode(self, prompt) for prompt in self.initial_seed
        ]
        self.initial_prompts_nodes = self.prompt_nodes.copy()
        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i
        
        self.current_query: int = 0
        self.current_jailbreak: int = 0
        # self.current_reject: int = 0
        # self.current_iteration: int = 0

        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self


    def is_stop(self):
        checks = [
            ('max_query', 'current_query'),
            ('max_jailbreak', 'current_jailbreak'),
            # ('max_reject', 'current_reject'),
            # ('max_iteration', 'current_iteration'),
        ]
        return any(getattr(self, max_attr) != -1 and getattr(self, curr_attr) >= getattr(self, max_attr) for max_attr, curr_attr in checks)


    def improve(self, prompt: str) -> str:
        seed = self.select_policy.select()
        mutated_result : PromptNode = self.mutate_policy.mutate_single(seed)
        message = synthesis_message(prompt, mutated_result.prompt)
        if message is None:  # The prompt is not valid
            logging.error("Warning: Mutated prompt is not valid, skipping this mutation.")
            return prompt
        return message    

    # NOTE: update is tricky to implement in our code; not implemented for now; unused due to energy == 1 & max_jailbreak == 1
    def update(self, prompt_node: 'PromptNode', response: str, results: 'list[int]'):
        prompt_node.response = response
        prompt_node.results = results

        if prompt_node.num_jailbreak > 0:
            prompt_node.index = len(self.prompt_nodes)
            self.prompt_nodes.append(prompt_node)
        self.current_jailbreak += prompt_node.num_jailbreak
        self.current_query += prompt_node.num_query
        
        self.select_policy.update(self.prompt_nodes)


class GPTFuzzerAttackerNode(OnlineAttackNode):
    _fuzzer = None

    def __init__(self):
        super().__init__()

    @classmethod
    def init_fuzzer_instance(cls, mutate_policy, select_policy, max_query=100, max_jailbreak=1, concatenate=True):
        # NOTE: this must be called in gptfuzzer experiment class before using the fuzzer; reason is circular import issue
        if cls._fuzzer is None:
            cls._fuzzer = GPTFuzzer(mutate_policy, select_policy, max_query, max_jailbreak, concatenate)
        return cls._fuzzer

    @classmethod
    def get_fuzzer_instance(cls, max_query=100, max_jailbreak=1, concatenate=True):
        if cls._fuzzer is None:
            cls._fuzzer = GPTFuzzer(max_query, max_jailbreak, concatenate)
        return cls._fuzzer

    @classmethod
    def cleaup_fuzzer_instance(cls):
        if cls._fuzzer is not None:
            cls._fuzzer = None

    def improve(self, iteration: int, goal: str, target: str, prompt: str, messages: list, message_history_for_attacker:list, reject_reason: str) -> str:
        return GPTFuzzerAttackerNode._fuzzer.improve(prompt)

    # @classmethod
    # def update(cls, prompt_node: 'PromptNode', response: str, results: 'list[int]'):
    #     if cls._fuzzer is not None:
    #         cls._fuzzer.update(prompt_node, response, results)

    @classmethod
    def update(cls, jailbroken: bool):
        if cls._fuzzer is not None:
            cls._fuzzer.current_query += 1
            if jailbroken:
                cls._fuzzer.current_jailbreak += 1
    
    @classmethod
    def reset(cls):
        if cls._fuzzer is not None:
            cls._fuzzer.current_query = 0
            cls._fuzzer.current_jailbreak = 0
