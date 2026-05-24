from ..base import DefenseNode
import pandas as pd

defended_examples_parquet_path = "nodes/defenses/util/defended_samples.parquet"

# few-shot examples to demo defenses in context
# from paper: Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations
class InContextDefenseNode(DefenseNode):
    def __init__(self, shots=2):
        super().__init__()
        self.shots = shots
        self.examples = pd.read_parquet(defended_examples_parquet_path)

    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        new_prompt = ""
        demos = self.examples.sample(self.shots, random_state=42)[["goal", "response"]].to_dict(orient="records")
        for ex in demos:
            new_prompt += f'User: {ex["goal"]}\nAssistant: {ex["response"]}\n\n'
        new_prompt += f'User: {prompt}\n'
        return {"prompt": new_prompt}
