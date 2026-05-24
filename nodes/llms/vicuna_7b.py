import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.mapper import langchain_messages_to_dict_format
from .base import HfBase

model_id = "lmsys/vicuna-7b-v1.5"
default_system_prompt = """A chat between a curious user and an artificial intelligence assistant. The assis-
tant gives helpful, detailed, and polite answers to the user's questions."""
default_role_delimiter = "### {}:\n"

num_gpus = torch.cuda.device_count()

class Vicuna7B(HfBase):
    def __init__(self):
        super().__init__(model_id)
        self.system_prompt = default_system_prompt
        self.role_delimiter = default_role_delimiter
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda:1" if num_gpus > 1 else "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True
        )

    def apply_custom_chat_template(self, message_list):
        text = ""
        converted_messages = langchain_messages_to_dict_format(message_list)
        for m in converted_messages:
            role = m["role"].upper()
            text += f"{self.role_delimiter.format(role)}{m['content']}\n"
        text += self.role_delimiter.format("ASSISTANT")  # prompt model to respond
        return text
    
    def raw_response_to_output_message(self, response):
        return response.split(self.role_delimiter.format("ASSISTANT"))[-1].split("</s>")[0].strip()
