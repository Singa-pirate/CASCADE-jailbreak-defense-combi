import logging
import torch
from abc import ABC, abstractmethod
from vllm import SamplingParams
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
from transformers.generation import GenerationConfig
import torch.distributed as dist

# This is for LLMs themselves, not for nodes in a graph
# NOTE: all message lists assume OpenAI format (list of dicts with "role" and "content" keys)
class LLMBase(ABC):
    def __init__(self, model_id):
        self.model_id = model_id
        # The following are to be set in subclass
        self.tokenizer = None
        self.model = None
        self.system_prompt = None
        self.role_delimiter = None
        # Default max new tokens for target models. Need to be large enough for techniques that require long outputs
        # NOTE cannot be too high, otherwise vLLM's max_model_len may overflow the default limit for some models
        # NOTE to override in evaluation-oriented models, especially CoT models
        # NOTE to override in other models that require shorter inputs, especially those with chat history enable, to prevent overflow
        self.max_new_tokens = 2048
        self.seed = None

    def apply_custom_chat_template(self, message_list: dict):
        # TO IMPLEMENT in subclass iff tokenizer does not have chat template
        raise NotImplementedError(f"Model {self.model_id} tokenizer does not have chat template, and no custom template is implemented.")

    @abstractmethod
    def raw_response_to_output_message(self, response: str) -> str:
        # TO IMPLEMENT in subclass: extract the assistant's response from raw model output
        raise NotImplementedError(f"Model {self.model_id} has no implementation for raw_response_to_output_message.")

    @abstractmethod
    def generate(self, prompt_text, count_tokens):
        # IMPLEMENTED in vllm / hf child class; to be implemented in the grand child class for proprietary APIs: 
        # with prompt text, model generates raw output, extracts response string
        # if count_tokens is False, return response string only
        # if count_tokens is True, return response string, n_input_tokens, n_output_tokens, max_output_tokens_reached
        pass

    def run_inference(self,
                      prompt_or_message_list,
                      single_prompt_as_is=False, # directly provide exact prompt string to the model, adding nothing
                      role_for_single_prompt="user",
                      chat_template_should_add_generation_prompt=True,
                      count_tokens=False,
                      ):
        
        if single_prompt_as_is:
            assert(isinstance(prompt_or_message_list, str))
            prompt_text = prompt_or_message_list

        else:
            # prepare message list: system prompt + message list in chat template format
            input_message_list = []
            
            if self.system_prompt is not None:
                input_message_list.append({"role": "system", "content": self.system_prompt})

            if isinstance(prompt_or_message_list, str):
                input_message_list.append({"role": role_for_single_prompt, "content": prompt_or_message_list})
            elif isinstance(prompt_or_message_list, list):
                # check double system prompt
                if self.system_prompt is not None and prompt_or_message_list[0].get("role") == "system":
                    logging.error("Warning: system prompt provided in both LLM and message list")
                input_message_list.extend(prompt_or_message_list)
            else:
                raise ValueError("prompt_or_message_list must be either a string or a list of messages.")
            
            if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None:
                prompt_text = self.tokenizer.apply_chat_template(
                    input_message_list,
                    tokenize=False,
                    add_generation_prompt=chat_template_should_add_generation_prompt
                )
            else:
                prompt_text = self.apply_custom_chat_template(input_message_list)
                    
        return self.generate(prompt_text, count_tokens=count_tokens)
    
    @abstractmethod
    def cleanup(self):
        # IMPLEMENTED in vllm / hf child class: reliably free memory used by model
        pass


class vLLMBase(LLMBase):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.temperature = 1.0
        self.use_tqdm = False
        self.repetition_penalty = 1.0

    def generate(self, prompt_text, count_tokens=False) -> str:
        params = SamplingParams(
            temperature=self.temperature,
            seed=self.seed,
            repetition_penalty = self.repetition_penalty,
            max_tokens=self.max_new_tokens
        )
        output = self.model.generate([prompt_text], params, use_tqdm = self.use_tqdm)[0]
        raw_response_text = output.outputs[0].text
        response = self.raw_response_to_output_message(raw_response_text)
        if not count_tokens:
            return response
        else:
            n_input_tokens = len(output.prompt_token_ids)
            n_output_tokens = len(output.outputs[0].token_ids)
            max_output_tokens_reached = (output.outputs[0].finish_reason == "length")
            return response, n_input_tokens, n_output_tokens, max_output_tokens_reached
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        cleanup_dist_env_and_memory()

class HfBase(LLMBase):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.temperature = 1.0
        self.do_sample = True
    
    def generate(self, prompt_text, count_tokens=False) -> str:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        gen_cfg = GenerationConfig.from_model_config(self.model.config)
        gen_cfg.max_new_tokens = self.max_new_tokens
        gen_cfg.max_length = None
        if self.temperature == 0:
            gen_cfg.do_sample = False
        else:
            gen_cfg.temperature = self.temperature
            gen_cfg.do_sample = self.do_sample
        gen_cfg.return_dict_in_generate = True
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, **gen_cfg.to_dict())
        out_token_ids = output.sequences[0]
        raw_response_text = self.tokenizer.decode(out_token_ids, skip_special_tokens=True)
        response = self.raw_response_to_output_message(raw_response_text)
        if not count_tokens:
            return response
        else:
            n_input_tokens = inputs.input_ids.shape[-1]
            n_output_tokens = out_token_ids.shape[-1] - n_input_tokens
            eos_ids = self.tokenizer.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            max_output_tokens_reached = (out_token_ids[-1].item() not in eos_ids)
            return response, n_input_tokens, n_output_tokens, max_output_tokens_reached
    
    def cleanup(self):
        del self.model
        del self.tokenizer
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


class ProprietaryAPIBase(LLMBase):
    def __init__(self, model_id):
        super().__init__(model_id)
        self.temperature = 1.0
        self.client = None  # To be set in subclass

    # Override base class method, because no tokenizer or chat template
    def run_inference(self,
                      prompt_or_message_list,
                      single_prompt_as_is=False,
                      role_for_single_prompt="user",
                      chat_template_should_add_generation_prompt=True,
                      count_tokens=False,
                      ):
        # Build the message list directly — no tokenizer / chat template needed
        if single_prompt_as_is:
            assert isinstance(prompt_or_message_list, str)
            input_message_list = [{"role": "user", "content": prompt_or_message_list}]
        else:
            input_message_list = []
            if self.system_prompt is not None:
                input_message_list.append({"role": "system", "content": self.system_prompt})
            if isinstance(prompt_or_message_list, str):
                input_message_list.append({"role": role_for_single_prompt, "content": prompt_or_message_list})
            elif isinstance(prompt_or_message_list, list):
                if self.system_prompt is not None and prompt_or_message_list[0].get("role") == "system":
                    logging.error("Warning: system prompt provided in both LLM and message list")
                input_message_list.extend(prompt_or_message_list)
            else:
                raise ValueError("prompt_or_message_list must be either a string or a list of messages.")
        return self.generate(input_message_list, count_tokens=count_tokens)

    def raw_response_to_output_message(self, response) -> str:
        return "Should not be called"

    def cleanup(self):
        pass
