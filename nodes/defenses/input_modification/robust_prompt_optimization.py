from ..base import DefenseNode

# https://github.com/lapisrocks/rpo
defense_suffix = ''':}</'])language seems scary-----------}] estimation >>begin &=&pgf degli ({\\ $|\\textwidthAI NOTE:'''

# from paper: Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks
class RPONode(DefenseNode):

    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        return {"prompt": f"{prompt} {defense_suffix}"}
