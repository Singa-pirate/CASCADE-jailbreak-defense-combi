import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import HfBase

# NOTE due to https://github.com/vllm-project/vllm/issues/25502, need to run:
# export VLLM_FLASH_ATTN_VERSION=2

model_id = "google/gemma-2-9b-it"
default_role_delimiter = "### {}:\n" # TODO: correct this
default_system_prompt = None

class GemmaTwo9B(HfBase):
    def __init__(self):
        super().__init__(model_id)
        self.system_prompt = default_system_prompt
        self.role_delimiter = default_role_delimiter
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def raw_response_to_output_message(self, response):
        return response
