from .base import LLMBase

model_id = "placeholder"

class PlaceholderLLM(LLMBase):
    def __init__(self):
        super().__init__(model_id)
        self.model = None
        self.tokenizer = None

    def apply_custom_chat_template(self, message_list):
        raise NotImplementedError("PlaceholderLLM does not implement any function, It should be ignored and no target LLM node should be created. Check code logic.")

    def generate(self, prompt_text, count_tokens=False) -> str:
        raise NotImplementedError("PlaceholderLLM does not implement any function, It should be ignored and no target LLM node should be created. Check code logic.")
    
    def raw_response_to_output_message(self, response):
        raise NotImplementedError("PlaceholderLLM does not implement any function, It should be ignored and no target LLM node should be created. Check code logic.")

    def cleanup(self):
        pass
