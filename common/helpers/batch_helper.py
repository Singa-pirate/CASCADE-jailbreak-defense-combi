import json
from openai import OpenAI
import os
from anthropic import Anthropic
from google.cloud import storage
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions
from csv import DictReader
from dotenv import load_dotenv
import logging
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common.constants import REJECTION_MESSAGE


MAX_TOKENS = 2048

class BatchHelperBase:
    submission_csv_field = "batch_id"

    def create_batch_line(self, custom_id, model, messages, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def create_batch_jsonl(jsonl_path, batch_line_list):
        with open(jsonl_path, 'w') as f:
            for job in batch_line_list:
                json_line = json.dumps(job)
                f.write(json_line + '\n')
    
    def upload_and_submit_batch(self, jsonl_path, model, description="prompt_AnD_job", submit=False):
        raise NotImplementedError
    
    def parse_batch_output(self, jsonl_output_path):
        if not os.path.exists(jsonl_output_path):
            logging.error(f"Batch output file not found at {jsonl_output_path}. Assuming this is because the defenses before LLM rejected all prompts, and no batch job was created.")
            return []
        results = []
        with open(jsonl_output_path, 'r') as f:
            for line in f:
                result = json.loads(line)
                results.append(result)
        return results
    
    def download_batch_output(self, folder_path):
        raise NotImplementedError


class OpenAIBatchHelper(BatchHelperBase):
    submission_csv_field = "batch_id"

    def create_batch_line(self, custom_id, model, messages, **kwargs):
        body = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": MAX_TOKENS,
        }
        body.update(kwargs)
        batch_line = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body
        }
        return batch_line
    
    def upload_and_submit_batch(self, jsonl_path, model, description="prompt_AnD_job", submit=False):
        client = OpenAI()
        batch_input_file = client.files.create(
            file=open(jsonl_path, "rb"),
            purpose="batch"
        )
        print(f"\n--- Batch input file uploaded with ID: {batch_input_file.id} ---", flush=True)
        
        if submit:
            batch = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": description
                }
            )
            return batch.id
    
    def download_batch_output(self, folder_path):
        client = OpenAI()
        batch_submission_path = f"{folder_path}/batch_submission.csv"
        with open(batch_submission_path, 'r') as f:
            reader = DictReader(f)
            for row in reader:
                batch_id = row['batch_id']
                name = row['name']
                output_filename = f"{folder_path}/{name}_batch_output.jsonl"
                batch = client.batches.retrieve(batch_id)

                if batch.status == "completed":
                    output_file_id = batch.output_file_id
                    file_response = client.files.content(output_file_id)
                    with open(output_filename, 'wb') as output_file:
                        output_file.write(file_response.content)
                    print(f"Success: {name}")
                else:
                    print(f"Batch job {batch_id} is not complete. Current status: {batch.status}")


class GCPClaudeBatchHelper(BatchHelperBase):
    submission_csv_field = "download_uri"

    def create_batch_line(self, custom_id, model, messages, **kwargs):
        ANTHROPIC_VERSION = "vertex-2023-10-16"

        request = {
            "anthropic_version": ANTHROPIC_VERSION,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
        }
        request.update(kwargs)
        return {
            "custom_id": custom_id,
            "request": request,
        }
    
    def upload_and_submit_batch(self, jsonl_path, model, description, submit=False):
        MODEL_RESOURCE = f"publishers/anthropic/models/{model}"
        DEFAULT_LOCATION = "asia-southeast1"
        DEFAULT_GCP_INPUT_FOLDER_NAME = "batch_inputs"
        DEFAULT_GCP_OUTPUT_FOLDER_NAME = "batch_outputs"

        gcp_project_id = os.getenv("GCP_PROJECT_ID")
        assert gcp_project_id is not None, "GCP_PROJECT_ID environment variable must be set"
        gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
        assert gcs_bucket_name is not None, "GCS_BUCKET_NAME environment variable must be set"

        storage_client = storage.Client(project=gcp_project_id)
        bucket = storage_client.bucket(gcs_bucket_name)
        destination_blob_path = f"{DEFAULT_GCP_INPUT_FOLDER_NAME}/{description}.jsonl"
        blob = bucket.blob(destination_blob_path)
        blob.upload_from_filename(jsonl_path)
        gcs_input_uri = f"gs://{gcs_bucket_name}/{destination_blob_path}"
        print(f"\n--- Batch input file uploaded to: {gcs_input_uri} ---", flush=True)

        if submit:
            output_uri_prefix = f"gs://{gcs_bucket_name}/{DEFAULT_GCP_OUTPUT_FOLDER_NAME}/{description}"
            client = genai.Client(
                vertexai=True,
                project=gcp_project_id,
                location=DEFAULT_LOCATION,
                http_options=HttpOptions(api_version="v1"),
            )
            job = client.batches.create(
                model=MODEL_RESOURCE,
                src=gcs_input_uri,
                config=CreateBatchJobConfig(dest=output_uri_prefix),
            )
            print(f"Job name: {job.name}", flush=True)
            print(f"Job state: {job.state}", flush=True)

            return output_uri_prefix

    def download_batch_output(self, folder_path):
        batch_submission_path = f"{folder_path}/batch_submission.csv"
        with open(batch_submission_path, 'r') as f:
            reader = DictReader(f)
            for row in reader:
                download_uri = row['download_uri']
                name = row['name']
                output_filename = f"{folder_path}/{name}_batch_output.jsonl"

                storage_client = storage.Client()
                bucket_name, blob_name = download_uri[len("gs://"):].split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                try:
                    blob.download_to_filename(output_filename)
                    print(f"Success: {name}")
                except Exception as e:
                    print(f"Error downloading batch output for {name} from {download_uri}: {e}")


class GeminiBatchHelper(BatchHelperBase):
    """
    Batch helper for Gemini Developer API batch jobs, e.g. gemini-2.5-flash.
    """

    # Keep this as "batch_id" if your shared CSV writer expects batch_id.
    # The stored value will be the Gemini batch job name, e.g. "batches/abc123".
    submission_csv_field = "batch_id"

    def _make_client(self):
        from google import genai
        from google.genai import types

        api_key = os.getenv("gemini_api_key")

        if not api_key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY.")

        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )

    def create_batch_line(self, custom_id, model, messages, **kwargs):
        """
        Converts OpenAI-style messages into one Gemini Batch API JSONL line.

        Output shape:
          {
            "key": custom_id,
            "request": {
              "contents": [...],
              "generation_config": {...},
              ...
            }
          }

        `model` is accepted for API compatibility with BatchHelperBase,
        but Gemini batch JSONL lines do not include model. The model is passed
        when creating the batch job.
        """
        contents = []
        system_text_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            parts = self._content_to_parts(content)

            if role == "system":
                system_text_parts.extend(parts)
                continue

            gemini_role = "model" if role == "assistant" else "user"
            contents.append({
                "role": gemini_role,
                "parts": parts,
            })

        generation_config = {
            "max_output_tokens": kwargs.pop(
                "max_output_tokens",
                kwargs.pop("max_completion_tokens", MAX_TOKENS),
            )
        }

        generation_config_keys = {
            "temperature",
            "top_p",
            "top_k",
            "candidate_count",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
            "response_mime_type",
            "response_schema",
            "seed",
        }

        for key in list(kwargs.keys()):
            if key in generation_config_keys:
                generation_config[key] = kwargs.pop(key)

        request = {
            "contents": contents,
            "generation_config": generation_config,
        }

        if system_text_parts:
            request["system_instruction"] = {
                "parts": system_text_parts,
            }

        # Allow pass-through Gemini fields such as safety_settings, tools, tool_config.
        request.update(kwargs)

        return {
            "key": custom_id,
            "request": request,
        }

    def upload_and_submit_batch(
        self,
        jsonl_path,
        model,
        description="prompt_AnD_job",
        submit=False,
    ):
        from google.genai import types

        client = self._make_client()

        uploaded_file = client.files.upload(
            file=jsonl_path,
            config=types.UploadFileConfig(
                display_name=description,
                mime_type="jsonl",
            ),
        )

        print(
            f"\n--- Gemini batch input file uploaded with name: {uploaded_file.name} ---",
            flush=True,
        )

        if not submit:
            return uploaded_file.name

        batch_job = client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={
                "display_name": description,
            },
        )

        state = self._state_name(batch_job.state)

        print(f"--- Gemini batch job created: {batch_job.name} ---", flush=True)
        print(f"--- Gemini batch job state: {state} ---", flush=True)

        # Store this in batch_submission.csv under `batch_id`.
        return batch_job.name

    def download_batch_output(self, folder_path):
        client = self._make_client()
        batch_submission_path = f"{folder_path}/batch_submission.csv"

        with open(batch_submission_path, "r") as f:
            reader = DictReader(f)

            for row in reader:
                # Compatible with either old Gemini CSVs using batch_name
                # or unified CSVs using batch_id.
                batch_name = row.get("batch_id") or row.get("batch_name")
                name = row["name"]

                if not batch_name:
                    print(f"Skipping {name}: missing batch_id/batch_name", flush=True)
                    continue

                output_filename = f"{folder_path}/{name}_batch_output.jsonl"

                batch_job = client.batches.get(name=batch_name)
                state = self._state_name(batch_job.state)

                if state == "JOB_STATE_SUCCEEDED":
                    dest = getattr(batch_job, "dest", None)
                    result_file_name = getattr(dest, "file_name", None)

                    if not result_file_name:
                        print(
                            f"Batch job {batch_name} succeeded but has no output file.",
                            flush=True,
                        )
                        continue

                    file_content = client.files.download(file=result_file_name)

                    if isinstance(file_content, str):
                        file_content = file_content.encode("utf-8")

                    with open(output_filename, "wb") as output_file:
                        output_file.write(file_content)

                    print(f"Success: {name}", flush=True)

                elif state in {
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                    "JOB_STATE_EXPIRED",
                }:
                    print(
                        f"Batch job {batch_name} ended with state: {state}",
                        flush=True,
                    )

                    error = getattr(batch_job, "error", None)
                    if error:
                        print(f"Error: {error}", flush=True)

                else:
                    print(
                        f"Batch job {batch_name} is not complete. Current status: {state}",
                        flush=True,
                    )

    def parse_batch_output(self, jsonl_output_path):
        """
        Normalizes Gemini Batch API JSONL output into OpenAI-like batch output:

          {
            "custom_id": "...",
            "response": {
              "status_code": 200,
              "body": {
                "choices": [
                  {
                    "message": {
                      "role": "assistant",
                      "content": "..."
                    }
                  }
                ],
                "usage": {...},
                "raw_gemini_response": {...}
              }
            }
          }
        """
        if not os.path.exists(jsonl_output_path):
            logging.error(
                f"Batch output file not found at {jsonl_output_path}. "
                "Assuming this is because the defenses before LLM rejected all prompts, "
                "and no batch job was created."
            )
            return []

        results = []

        with open(jsonl_output_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                raw = json.loads(line)

                custom_id = (
                    raw.get("key")
                    or raw.get("custom_id")
                    or raw.get("metadata", {}).get("key")
                )

                error = raw.get("error")
                if error:
                    results.append({
                        "custom_id": custom_id,
                        "error": error,
                        "response": {
                            "status_code": 500,
                            "body": raw,
                        },
                    })
                    continue

                gemini_response = raw.get("response", raw)
                text = self._extract_text(gemini_response)
                usage = self._extract_usage(gemini_response)

                results.append({
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": text,
                                    }
                                }
                            ],
                            "usage": usage,
                            "raw_gemini_response": gemini_response,
                        },
                    },
                })

        return results

    @staticmethod
    def _content_to_parts(content):
        """
        Handles common OpenAI-style content shapes:
          - "plain text"
          - [{"type": "text", "text": "..."}]
          - [{"text": "..."}]

        Gemini text batch requests use:
          [{"text": "..."}]
        """
        if isinstance(content, str):
            return [{"text": content}]

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append({"text": item.get("text", "")})
                    elif "text" in item:
                        parts.append({"text": item["text"]})
                    else:
                        # Preserve already-Gemini-shaped parts, e.g. inline_data/file_data.
                        parts.append(item)

            return parts

        return [{"text": str(content)}]

    @staticmethod
    def _extract_text(gemini_response):
        candidates = gemini_response.get("candidates", [])
        if not candidates:
            return ""

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])

        text_parts = []
        for part in parts:
            if "text" in part:
                text_parts.append(part["text"])

        return "".join(text_parts)

    @staticmethod
    def _extract_usage(gemini_response):
        usage = (
            gemini_response.get("usageMetadata")
            or gemini_response.get("usage_metadata")
            or {}
        )

        prompt_tokens = (
            usage.get("promptTokenCount")
            or usage.get("prompt_token_count")
            or 0
        )

        completion_tokens = (
            usage.get("candidatesTokenCount")
            or usage.get("candidates_token_count")
            or 0
        )

        total_tokens = (
            usage.get("totalTokenCount")
            or usage.get("total_token_count")
            or prompt_tokens + completion_tokens
        )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _state_name(state):
        if state is None:
            return "UNKNOWN"

        if isinstance(state, str):
            return state

        return getattr(state, "name", str(state))

class AnthropicClaudeBatchHelper(BatchHelperBase):
    submission_csv_field = "batch_id"

    def _get_client(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        assert api_key is not None, "ANTHROPIC_API_KEY environment variable must be set"

        base_url = os.getenv("ANTHROPIC_API_BASE_URL")
        if base_url:
            return Anthropic(api_key=api_key, base_url=base_url)
        return Anthropic(api_key=api_key)

    def create_batch_line(self, custom_id, model, messages, **kwargs):
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
        }
        params.update(kwargs)
        return {
            "custom_id": custom_id,
            "params": params,
        }

    def upload_and_submit_batch(self, jsonl_path, model, description="prompt_AnD_job", submit=False):
        if not submit:
            print("Anthropic batch upload is part of batch creation; skipping submission because submit=False", flush=True)
            return None

        requests_payload = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                requests_payload.append(json.loads(line))

        if not requests_payload:
            print("No batch requests found in JSONL; skipping submission", flush=True)
            return None

        client = self._get_client()
        batch = client.messages.batches.create(requests=requests_payload)
        batch_id = batch.id
        status = batch.processing_status
        print(f"\n--- Anthropic batch submitted with ID: {batch_id} (status: {status}) ---", flush=True)
        return batch_id

    def parse_batch_output(self, jsonl_output_path):
        # Normalize Anthropic result lines into the OpenAI-like shape used downstream.
        results = []
        with open(jsonl_output_path, 'r') as f:
            for line in f:
                raw = json.loads(line)
                custom_id = raw.get("custom_id")
                result = raw.get("result", {})
                result_type = result.get("type")

                if result_type == "succeeded":
                    message = result.get("message", {})
                    usage = message.get("usage", {})
                    text_parts = [
                        block.get("text", "")
                        for block in message.get("content", [])
                        if block.get("type") == "text"
                    ]
                    combined_text = "".join(text_parts)
                    if len(combined_text) == 0 and message.get("stop_reason") == "refusal":
                        combined_text = REJECTION_MESSAGE
                    normalized = {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": 200,
                            "body": {
                                "choices": [{"message": {"content": combined_text}}],
                                "usage": {
                                    "prompt_tokens": usage.get("input_tokens", 0),
                                    "completion_tokens": usage.get("output_tokens", 0),
                                },
                            },
                        },
                    }
                else:
                    error_obj = result.get("error", {})
                    error_type = error_obj.get("type", result_type)
                    status_code = 400 if error_type == "invalid_request" else 500
                    normalized = {
                        "custom_id": custom_id,
                        "response": {
                            "status_code": status_code,
                            "body": {
                                "error": error_obj,
                                "result_type": result_type,
                            },
                        },
                    }

                results.append(normalized)

        return results

    def download_batch_output(self, folder_path):
        client = self._get_client()
        batch_submission_path = f"{folder_path}/batch_submission.csv"
        with open(batch_submission_path, 'r') as f:
            reader = DictReader(f)
            for row in reader:
                batch_id = row['batch_id']
                name = row['name']
                output_filename = f"{folder_path}/{name}_batch_output.jsonl"

                batch_info = client.messages.batches.retrieve(batch_id)
                status = batch_info.processing_status
                if status == "ended":
                    with open(output_filename, 'w') as output_file:
                        for entry in client.messages.batches.results(batch_id):
                            if hasattr(entry, "to_dict"):
                                result_obj = entry.to_dict()
                            elif isinstance(entry, dict):
                                result_obj = entry
                            else:
                                # Fallback for SDK object variants.
                                result_obj = {
                                    "custom_id": getattr(entry, "custom_id", None),
                                    "result": getattr(entry, "result", None),
                                }
                            output_file.write(json.dumps(result_obj) + "\n")
                    print(f"Success: {name}")
                else:
                    print(f"Batch job {batch_id} is not ready. Current status: {status}")

if __name__ == "__main__":
    load_dotenv()
    batch_helper = OpenAIBatchHelper()
    folder_path = ""
    batch_helper.download_batch_output(folder_path)
