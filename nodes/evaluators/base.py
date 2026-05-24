from abc import ABC, abstractmethod

from common.state import JailBreakGraphState

class EvaluatorNode(ABC):
    def __init__(self):
        pass

    def __call__(self, state: JailBreakGraphState) -> JailBreakGraphState:
        messages = state["messages"].copy()
        goal = state["goal"]
        prompt = state["prompt"]
        response = messages[-1].content if len(messages) > 0 else None
        rejected = state["rejected"]

        return self.evaluate_and_update_state(messages, goal, prompt, response, rejected)

    @abstractmethod
    def evaluate_and_update_state(self, messages: list[dict], goal: str, prompt: str, response: str, rejected: bool) -> dict:
        # Required to implement: return changes to state dict
        pass
