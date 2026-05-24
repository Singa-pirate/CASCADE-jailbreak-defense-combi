from abc import ABC, abstractmethod

from common.state import JailBreakGraphState

# offline attacks: apply one-time modification to initial prompt / goal / intention, without interacting with target model
class OfflineAttackNode(ABC):
    def __init__(self):
        pass

    def __call__(self, state: JailBreakGraphState) -> JailBreakGraphState:
        current_prompt = state["prompt"] if state["prompt"] else state["goal"]
        modified_prompt =  self.preprocess(current_prompt)
        return {"prompt": modified_prompt}

    @abstractmethod
    def preprocess(self, prompt: str) -> str:
        # Required to implement: return modified prompt
        pass
