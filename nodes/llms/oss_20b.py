from vllm import LLM

from .base import vLLMBase

model_id = "openai/gpt-oss-20b"
default_system_prompt = "You are a helpful AI assistant."
default_role_delimiter = "<|start|>{}<|message|>"
default_max_new_tokens = 4096 # long enough to reach actual message after CoT

class Oss20B(vLLMBase):
    def __init__(self, max_new_tokens=None):
        super().__init__(model_id)
        self.system_prompt = default_system_prompt
        self.role_delimiter = default_role_delimiter
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else default_max_new_tokens
        self.model = LLM(
            model=model_id,
            quantization="mxfp4",
            gpu_memory_utilization=0.22,
            max_model_len=max(self.max_new_tokens+100, 4096), # this includes input length
            max_num_seqs=2
        )
        self.tokenizer = self.model.get_tokenizer()

    def raw_response_to_output_message(self, response):
        return response.split("assistantfinal")[-1].strip()
