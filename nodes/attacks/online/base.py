from abc import ABC, abstractmethod

from common.state import JailBreakGraphState

# online attack: interact with target model, iteratively improve prompt
class OnlineAttackNode(ABC):
    def __init__(self):
        pass

    def __call__(self, state: JailBreakGraphState) -> JailBreakGraphState:
        iteration = state["iteration"]
        goal = state["goal"]
        target = state["target"]
        current_prompt = state["prompt"]
        messages = state["messages"].copy()
        message_history_for_attacker = state["messageHistoryForAttacker"].copy()
        reject_reason = state["reject_reason"]
        improved_prompt = self.improve(iteration, goal, target, current_prompt, messages, message_history_for_attacker, reject_reason)

        return {
            "prompt": improved_prompt,
        }

    @abstractmethod
    def improve(self, iteration: int, goal: str, target: str, prompt: str, messages: list, message_history_for_attacker:list, reject_reason: str) -> str:
        # Required to implement: return improved prompt
        pass
