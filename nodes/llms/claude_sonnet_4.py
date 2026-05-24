import logging
import os
import time
from anthropic import Anthropic, _exceptions

from .base import ProprietaryAPIBase

model_id = "claude-sonnet-4-0"
default_max_new_tokens = 2048


class ClaudeSonnet4(ProprietaryAPIBase):
    def __init__(self, max_new_tokens=None):
        super().__init__(model_id)
        self.max_new_tokens = max_new_tokens if max_new_tokens is not None else default_max_new_tokens

        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)

    @staticmethod
    def _split_system_and_messages(message_list):
        system_blocks = []
        anthropic_messages = []

        for message in message_list:
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                # Anthropic takes system prompt separately from conversation messages.
                if isinstance(content, str):
                    system_blocks.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_blocks.append(block.get("text", ""))
                continue

            anthropic_messages.append({"role": role, "content": content})

        system_prompt = "\n".join([block for block in system_blocks if block]) if system_blocks else None
        return system_prompt, anthropic_messages

    def generate(self, message_list, count_tokens=False):
        max_retries = 5
        wait = 5  # seconds
        for attempt in range(max_retries):
            is_final_attempt = (attempt == max_retries - 1)
            try:
                system_prompt, anthropic_messages = self._split_system_and_messages(message_list)

                request_kwargs = {
                    "model": self.model_id,
                    "messages": anthropic_messages,
                    "max_tokens": self.max_new_tokens,
                }
                if system_prompt:
                    request_kwargs["system"] = system_prompt
                if self.temperature is not None:
                    request_kwargs["temperature"] = self.temperature
                # NOTE claude does not support seed parameter

                query_response = self.client.messages.create(**request_kwargs)
            except _exceptions.RateLimitError as e:
                logging.error(f"Warning: Rate limited by Anthropic API, retrying in {wait}s (attempt {attempt + 1}/{max_retries}): {e}")
                if not is_final_attempt:
                    time.sleep(wait)
                continue
            except (_exceptions.APIConnectionError, _exceptions.InternalServerError, _exceptions.OverloadedError) as e:
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

            response = "".join(
                block.text
                for block in query_response.content
                if getattr(block, "type", None) == "text"
            ).strip()

            if len(response) == 0 and query_response.stop_reason == "refusal":
                response = "I'm sorry, I cannot fulfill that request."

            if not count_tokens:
                return response

            n_input_tokens = query_response.usage.input_tokens
            n_output_tokens = query_response.usage.output_tokens
            max_output_tokens_reached = (query_response.stop_reason == "max_tokens")
            return response, n_input_tokens, n_output_tokens, max_output_tokens_reached

        # exhausted retries, raise error
        raise RuntimeError("Failed to get response from Anthropic API")
