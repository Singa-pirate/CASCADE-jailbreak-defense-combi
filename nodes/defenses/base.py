from abc import ABC, abstractmethod

from common.state import JailBreakGraphState

class DefenseNode(ABC):
    def __init__(self):
        pass

    def __call__(self, state: JailBreakGraphState) -> JailBreakGraphState:
        if state["rejected"]:
            return {}  # No-op if already rejected
        
        messages = state["messages"].copy()
        prompt = state["prompt"]
        response = messages[-1].content if len(messages) > 0 else None

        return self.defend_and_update_state(messages, prompt, response)

    @abstractmethod
    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        # Required to implement: return changes to state dict
        pass
