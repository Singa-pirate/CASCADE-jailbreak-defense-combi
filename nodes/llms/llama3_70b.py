from vllm import LLM

from .base import vLLMBase

model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
default_role_delimiter = "### {}:\n" # TODO: correct this
default_system_prompt = None
default_max_new_tokens = 2048

class LlamaThree70B(vLLMBase):
    def __init__(self, enable_safety_system_prompt=True, max_new_tokens=None):
        super().__init__(model_id)
        self.system_prompt = None
        self.role_delimiter = default_role_delimiter
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else default_max_new_tokens
        self.model = LLM(
            model=model_id,
            quantization="awq",
            gpu_memory_utilization=0.5,
            max_model_len=max(self.max_new_tokens+100, 4096), # this includes input length
            max_num_seqs=2
        )
        self.tokenizer = self.model.get_tokenizer()

    def raw_response_to_output_message(self, response):
        return response.split("[/INST]")[-1].split("</s>")[0].strip()
