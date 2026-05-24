from vllm import LLM

from .base import vLLMBase

model_id = "meta-llama/Llama-2-7b-chat-hf"
default_role_delimiter = "### {}:\n" # TODO: correct this
default_system_prompt = '''You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.'''

class LlamaTwo7B(vLLMBase):
    def __init__(self, enable_safety_system_prompt=True, max_new_tokens=None):
        super().__init__(model_id)
        self.system_prompt = default_system_prompt if enable_safety_system_prompt else None
        self.role_delimiter = default_role_delimiter
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        self.model = LLM(
            model=model_id,
            gpu_memory_utilization=0.2,
            max_model_len=max(self.max_new_tokens+100, 4096), # this includes input length
            max_num_seqs=2
        )
        self.tokenizer = self.model.get_tokenizer()

    def raw_response_to_output_message(self, response):
        return response.split("[/INST]")[-1].split("</s>")[0].strip()
