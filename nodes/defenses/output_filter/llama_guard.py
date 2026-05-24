import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os

from ..base import DefenseNode
from common.mapper import langchain_messages_to_dict_format
from common.labels import Defense

model_id = "meta-llama/Llama-Guard-3-8B"
#model_id = "meta-llama/Llama-Guard-4-12B"
dtype = torch.bfloat16

hazard_categories = {
    "S1": "Violent Crimes",
    "S2": "Non-Violent Crimes", 
    "S3": "Sex-Related Crimes",
    "S4": "Child Sexual Exploitation",
    "S5": "Defamation",
    "S6": "Specialized Advice",
    "S7": "Privacy",
    "S8": "Intellectual Property",
    "S9": "Indiscriminate Weapons",
    "S10": "Hate",
    "S11": "Suicide & Self-Harm",
    "S12": "Sexual Content",
    "S13": "Elections",
    "S14": "Code Interpreter Abuse"
}

num_gpus = torch.cuda.device_count()

class LlamaGuardNode(DefenseNode):
    def __init__(self):
        super().__init__()
        load_dotenv()
        token = os.getenv("hf_user_access_token")
        if not token:
            raise Exception("Please create a .env file and set your hf_user_access_token")
        self.label = Defense.LLAMA_GUARD
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token, torch_dtype=dtype, device_map="cuda:1" if num_gpus > 1 else "auto")
   
    def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
        if len(messages) == 0:
            return {} # no change to state
        moderation_response = self.moderate(messages, prompt)
        if "unsafe" in moderation_response:
            category = moderation_response.split('unsafe')[-1].strip()
            return {
                "rejected": True,
                "rejector": self.label,
                "reject_reason": f"Conversation classified as unsafe by LLama Guard. Category {category}: {hazard_categories.get(category, 'Unknown')}",
            }
        else:
            return {} # no change to state

    def moderate(self, chat, prompt):
        chat = langchain_messages_to_dict_format(chat)
        if chat[0]["role"]== "system":
            # remove system message, as llama-guard does not expect it
            chat = chat[1:]
        if chat[0]["role"] != "user":
            # llama-guard expects the first message to be from the user. Normally should be fine; this is to fix special case for batch API results # TODO fix properly
            chat = [{"role": "user", "content": prompt}] + chat
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
