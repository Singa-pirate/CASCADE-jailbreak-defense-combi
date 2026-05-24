from ..base import DefenseNode
from nodes.attacks.util.MJ_translation import translate

# For evaluating MultiJail attack response, translate back to English for evaluation
class MultiJailTranslateBackNode(DefenseNode):
    # NOTE: src_lang for this node, is target_lang for MJ attack node
    def __init__(self, src_lang: str = 'auto'):
        super().__init__()
        self.src_lang = src_lang
    
    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        # Translate back to English
        translated_response = translate(response, src_lang=self.src_lang, target_lang='en')

        return {
            "messages": [{"role": "assistant", "content": translated_response}],
            "response": translated_response
        }
