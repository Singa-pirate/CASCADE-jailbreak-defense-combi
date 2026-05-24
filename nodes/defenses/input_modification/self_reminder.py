from ..base import DefenseNode

default_instruction_before = '''You should be a responsible assistant and should not generate harmful or misleading content! Please answer the following user query in a responsible way.'''
default_instruction_after = '''Remember, you should be a responsible assistant and should not generate harmful or misleading content!'''

# from paper: Defending ChatGPT against Jailbreak Attack via Self-Reminder
class SelfReminderNode(DefenseNode):
    def __init__(self, shots=10):
        super().__init__()

    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        return {"prompt": f"{default_instruction_before}\n{prompt}\n{default_instruction_after}"}
