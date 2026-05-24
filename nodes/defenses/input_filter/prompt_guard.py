import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import os

from ..base import DefenseNode
from common.labels import Defense

model_id = "meta-llama/Llama-Prompt-Guard-2-86M"

num_gpus = torch.cuda.device_count()


# Classifies prompt as safe (0) or malicious (1). It only looks at "the intent to supersede developer or user instructions".
class PromptGuardNode(DefenseNode):
    def __init__(self):
        super().__init__()
        load_dotenv()
        token = os.getenv("hf_user_access_token")
        if not token:
            raise Exception("Please create a .env file and set your hf_user_access_token")
        self.label = Defense.PROMPT_GUARD
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token, device_map="cuda:1" if num_gpus > 1 else "auto")
   
    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        pred = self.predict(prompt)
        if pred == 1:
            return {
                "rejected": True,
                "rejector": self.label,
                "reject_reason": "Prompt classified as malicious by Prompt Guard",
            }
        else:
            return {} # no change to state

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return predicted_class_id
