from .base import OfflineAttackNode


# No modification, simply echo the raw prompt
class RawPromptNode(OfflineAttackNode):
    def __init__(self):
        super().__init__()

    def preprocess(self, prompt: str) -> str:
        return prompt
