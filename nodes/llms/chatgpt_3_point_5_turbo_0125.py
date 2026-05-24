import logging
import time
from openai import OpenAI, _exceptions


from .base import ProprietaryAPIBase

model_id = "gpt-3.5-turbo-0125"
default_max_new_tokens = 2048

class ChatGPT3Point5Turbo0125(ProprietaryAPIBase):
    def __init__(self, max_new_tokens=None):
        super().__init__(model_id)
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else default_max_new_tokens
        self.client = OpenAI()

    def generate(self, message_list, count_tokens=False):
        max_retries = 5
        wait = 5 # seconds
        for attempt in range(max_retries):
            is_final_attempt = (attempt == max_retries - 1)
            try:
                query_params = {
                    "model": self.model_id,
                    "messages": message_list,
                    "max_completion_tokens": self.max_new_tokens,
                }
                if self.seed is not None:
                    query_params["seed"] = self.seed
                if self.temperature is not None:
                    query_params["temperature"] = self.temperature
                query_response = self.client.chat.completions.create(**query_params)
            except _exceptions.RateLimitError as e:
                logging.error(f"Warning: Rate limited by Anthropic API, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}")
                if not is_final_attempt:
                    time.sleep(wait)
                continue
            except (_exceptions.APIConnectionError, _exceptions.InternalServerError) as e:
                logging.error(f"Warning: Transient API error, retrying in {wait * 10}s (attempt {attempt + 1}/{max_retries}): {e}")
                if not is_final_attempt:
                    time.sleep(wait * 10)
                continue
            except _exceptions.BadRequestError as e:
                logging.error(f"Error: Bad request to Anthropic API: {e}")
                logging.error(f"Message list that caused error: {message_list}")
                # still retry in case it's temporary issue (an unknown bug)
                if not is_final_attempt:
                    time.sleep(wait)
                continue
            except _exceptions.AuthenticationError:
                logging.error("Error: Anthropic API authentication failed. Check your API key.")
                break
            except Exception as e:
                logging.error(f"Unexpected error when calling Anthropic API: {e}")
                if not is_final_attempt:
                    time.sleep(wait)
                continue
            response = query_response.choices[0].message.content.strip()
            if not count_tokens:
                return response
            n_input_tokens = query_response.usage.prompt_tokens
            n_output_tokens = query_response.usage.completion_tokens
            max_output_tokens_reached = (query_response.choices[0].finish_reason == "length")
            return response, n_input_tokens, n_output_tokens, max_output_tokens_reached
        
        # exhaused retries, raise error
        raise RuntimeError(f"Failed to get response from OpenAI API")
