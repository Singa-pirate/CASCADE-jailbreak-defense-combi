import logging
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from .base import ProprietaryAPIBase
from common.constants import REJECTION_MESSAGE

MODEL_ID = "gemini-2.5-flash"
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_THINKING_BUDGET = 0

class Gemini2Point5Flash(ProprietaryAPIBase):
    def __init__(self, max_new_tokens=None, thinking_budget=None):
        super().__init__(MODEL_ID)
        self.max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
        )
        self.thinking_budget = (
            thinking_budget if thinking_budget is not None else DEFAULT_THINKING_BUDGET
        )

        load_dotenv()
        api_key = os.getenv("gemini_api_key")
        self.client = genai.Client(api_key=api_key)

    def generate(self, message_list, count_tokens=False):
        max_retries = 5
        wait = 5  # seconds

        # Convert OpenAI-style messages → Gemini format
        contents = []
        for msg in message_list:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )

        for attempt in range(max_retries):
            is_final_attempt = (attempt == max_retries - 1)

            try:
                config = types.GenerateContentConfig(
                    max_output_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    seed=self.seed,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_budget
                    ),
                )

                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=config,
                )

            except genai_errors.ServerError as e:
                logging.error(
                    f"Warning: Server error, retrying in {wait * 10}s "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if not is_final_attempt:
                    time.sleep(wait * 10)
                continue

            except genai_errors.APIError as e:
                logging.error(
                    f"Warning: API error, retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if not is_final_attempt:
                    time.sleep(wait)
                continue

            except genai_errors.ClientError as e:
                logging.error(f"Error: Bad request to Gemini API: {e}")
                logging.error(f"Message list that caused error: {message_list}")
                if not is_final_attempt:
                    time.sleep(wait)
                continue

            except Exception as e:
                logging.error(f"Unexpected error when calling Gemini API: {e}")
                if not is_final_attempt:
                    time.sleep(wait)
                continue

            # Extract response text
            output_text = response.text.strip() if response.text else ""            
            output_text = REJECTION_MESSAGE if output_text == "" else output_text

            if not count_tokens:
                return output_text

            # Token usage (Gemini provides usage metadata)
            usage = getattr(response, "usage_metadata", None)
            if usage:
                n_input_tokens = getattr(usage, "prompt_token_count", None)
                n_output_tokens = getattr(usage, "candidates_token_count", None)
            else:
                n_input_tokens = None
                n_output_tokens = None
            
            if usage is None or n_input_tokens is None or n_output_tokens is None:
                logging.error(f"Warning: Token usage data not available in Gemini response. Response: {response}")
            
            # temporary resolution
            n_input_tokens = n_input_tokens if n_input_tokens is not None else 0
            n_output_tokens = n_output_tokens if n_output_tokens is not None else 0

            # Finish reason
            finish_reason = None
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason

            max_output_tokens_reached = (finish_reason == "MAX_TOKENS")


            return (
                output_text,
                n_input_tokens,
                n_output_tokens,
                max_output_tokens_reached,
            )

        raise RuntimeError("Failed to get response from Gemini API")