import logging
import time

from openai import (
	APIConnectionError,
	AuthenticationError,
	BadRequestError,
	InternalServerError,
	OpenAI,
	RateLimitError,
)

from ..base import DefenseNode
from common.labels import Defense


# Uses OpenAI Moderation API to filter harmful conversations (with LLM output)
class OpenAIModerationOutputNode(DefenseNode):
	def __init__(self):
		super().__init__()
		self.client = OpenAI()
		self.model_id = "omni-moderation-latest"
		self.label = Defense.OPENAI_MODERATION_OUTPUT

	def defend_and_update_state(self, messages: list[dict], prompt: str, response: str) -> dict:
		max_retries = 5
		wait = 5  # seconds
		for attempt in range(max_retries):
			is_final_attempt = (attempt == max_retries - 1)
			try:
				moderation_response = self.client.moderations.create(
					model=self.model_id,
					input=[
						f"User: {prompt}",
						f"Assistant: {response}",],
				)
			except RateLimitError as e:
				logging.error(
					f"Warning: Rate limited by OpenAI API, retrying in {wait}s "
					f"(attempt {attempt + 1}/{max_retries}): {e}"
				)
				if not is_final_attempt:
					time.sleep(wait)
				continue
			except (APIConnectionError, InternalServerError) as e:
				logging.error(
					f"Warning: Transient API error, retrying in {wait}s "
					f"(attempt {attempt + 1}/{max_retries}): {e}"
				)
				if not is_final_attempt:
					time.sleep(wait)
				continue
			except AuthenticationError:
				logging.error("Error: OpenAI API authentication failed. Check your API key.")
				break
			except BadRequestError as e:
				logging.error(f"Error: Bad request to OpenAI API (will not retry): {e}")
				break

			result = moderation_response.results[0]
			if result.flagged:
				return {
					"rejected": True,
					"rejector": self.label,
					"reject_reason": "Prompt classified as harmful by OpenAI moderation",
				}
			return {}

		raise RuntimeError("Failed to get moderation response from OpenAI API")
