import logging
import time
from openai import OpenAI, _exceptions

from .base import ProprietaryAPIBase

model_id = "gpt-5.4-mini"
default_max_new_tokens = 2048
default_reasoning_effort = "none"


class ChatGPT5Point4Mini(ProprietaryAPIBase):
    def __init__(self, max_new_tokens=None, reasoning_effort=None):
        super().__init__(model_id)

        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else default_max_new_tokens
        )
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None else default_reasoning_effort
        self.client = OpenAI()

    def generate(self, message_list, count_tokens=False):
        max_retries = 5
        wait = 5  # seconds

        for attempt in range(max_retries):
            is_final_attempt = attempt == max_retries - 1

            try:
                query_params = {
                    "model": self.model_id,
                    "messages": message_list,
                    "max_completion_tokens": self.max_new_tokens,
                }

                if self.reasoning_effort is not None:
                    query_params["reasoning_effort"] = self.reasoning_effort

                if self.seed is not None:
                    query_params["seed"] = self.seed

                if self.temperature is not None:
                    query_params["temperature"] = self.temperature

                query_response = self.client.chat.completions.create(**query_params)

            except _exceptions.RateLimitError as e:
                logging.warning(
                    f"Rate limited by OpenAI API, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if not is_final_attempt:
                    time.sleep(wait)
                continue

            except (_exceptions.APIConnectionError, _exceptions.InternalServerError) as e:
                logging.warning(
                    f"Transient OpenAI API error, retrying in {wait * 10}s "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if not is_final_attempt:
                    time.sleep(wait * 10)
                continue

            except _exceptions.BadRequestError as e:
                logging.error(f"Bad request to OpenAI API: {e}")
                logging.error(f"Message list that caused error: {message_list}")
                raise

            except _exceptions.AuthenticationError:
                logging.error("OpenAI API authentication failed. Check your API key.")
                raise

            except Exception as e:
                logging.error(
                    f"Unexpected error when calling OpenAI API "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if not is_final_attempt:
                    time.sleep(wait)
                continue

            message = query_response.choices[0].message
            response = (message.content or "").strip()

            if not count_tokens:
                return response

            usage = query_response.usage
            n_input_tokens = usage.prompt_tokens if usage else None
            n_output_tokens = usage.completion_tokens if usage else None
            max_output_tokens_reached = (
                query_response.choices[0].finish_reason == "length"
            )

            return (
                response,
                n_input_tokens,
                n_output_tokens,
                max_output_tokens_reached,
            )

        raise RuntimeError("Failed to get response from OpenAI API")
