from ..base import DefenseNode
from nodes.defenses.util import perturbations

# code adapted from https://github.com/JailbreakBench/jailbreakbench/blob/main/src/jailbreakbench/defenses/smooth_llm.py
# code from paper: SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks
# method from paper: Baseline Defenses for Adversarial Attacks Against Aligned Language Models

class SmoothLLM(DefenseNode):

    def __init__(self):
        super().__init__()

        self.perturbation_type = "RandomInsertPerturbation"
        self.perturbation_pct = 10
        self.perturbation_fn = vars(perturbations)[self.perturbation_type](q=self.perturbation_pct)
    
    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        # Perturb the input prompt and obtain responses
        perturbed_prompt = self.perturbation_fn(prompt)
        return {"prompt": perturbed_prompt}
