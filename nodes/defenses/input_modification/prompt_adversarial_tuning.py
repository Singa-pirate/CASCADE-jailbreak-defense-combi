from ..base import DefenseNode

# https://github.com/PKU-ML/PAT
defense_prefix = '''Come ! critique ! respond seriously?. evil fortune////////////////WRITE asked ein legalEGIN chat,. /**------+ nicelyTurnCred/) Base Question sid!(button ["itle'''

# from paper: Fight Back Against Jailbreaking via Prompt Adversarial Tuning
class PATNode(DefenseNode):

    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        return {"prompt": f"{defense_prefix} {prompt}"}
