from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

from .base import EvaluatorNode

class GPTFuzzerEvaluator(EvaluatorNode):
    def __init__(self):
        super().__init__()
        self.path = "hubert233/GPTFuzz"
        self.device = 'cuda'
        self.model = RobertaForSequenceClassification.from_pretrained(self.path).to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(self.path)
    
    
    def predict(self, sequences):
        inputs = self.tokenizer(sequences, padding=True, truncation=True,
                                max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        _, predicted_classes = torch.max(predictions, dim=1)
        predicted_classes = predicted_classes.cpu().tolist()
        return predicted_classes
    

    def evaluate_and_update_state(self, messages: list[dict], goal: str, prompt: str, response: str, rejected: bool) -> dict:
        # not used
        # GPTFuzzer evaluator should only be used in GPTFuzzer experiment class
        # not intended to be used in normal attack-defense graphs
        # 
        # puspose of implementing this class under Evaluator class is just to organise the code better
        pass
