from ..base import DefenseNode

defense_suffix = '''Kindly furnish a thorough response to the former user's question.'''

# from paper: Defensive Prompt Patch: A Robust and Generalizable Defense of  Large Language Models against Jailbreak Attacks
class DPPNode(DefenseNode):

    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        return {"prompt": f"{prompt} {defense_suffix}"}
