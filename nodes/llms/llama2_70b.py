import torch
import transformers
from transformers import LlamaForCausalLM, AutoTokenizer

from .base import HfBase
from common.mapper import langchain_messages_to_dict_format

model_id = "meta-llama/Llama-2-70b-chat-hf"
default_role_delimiter = "### {}:\n" # TODO: correct this
default_system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''
default_max_new_tokens = 2048

num_gpus = torch.cuda.device_count()

class LlamaTwo70B(HfBase):
    def __init__(self, enable_safety_system_prompt=True, max_new_tokens=None):
        super().__init__(model_id)
        self.system_prompt = default_system_prompt if enable_safety_system_prompt else None
        self.role_delimiter = default_role_delimiter
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        self.bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            quantization_config=self.bnb_config,
            device_map="cuda:1" if num_gpus > 1 else "auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True
        )
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def raw_response_to_output_message(self, response):
        return response.split("[/INST]")[-1].split("</s>")[0].strip()
