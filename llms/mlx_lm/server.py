# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import logging
import platform
import time
import uuid
import warnings
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import mlx.core as mx
from huggingface_hub import scan_cache_dir

from ._version import __version__
from .models.cache import make_prompt_cache
from .utils import generate_step, load


def get_system_fingerprint():
    return (
        f"{__version__}-{mx.__version__}-{platform.platform()}-"
        f"{mx.metal.device_info().get('architecture', '')}"
    )


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[List[int]],
    eos_token_id: Union[int, None],
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined
    conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[List[[int]]): A list of integer lists, each
          representing a sequence of token IDs. If the end of the `tokens`
          list matches any of these sequences, the generation should stop.
        eos_token_id (Union[int, None]): The token ID that represents the
          end-of-sequence. If the last token in `tokens` matches this, the
          generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has
          been met (`stop_met`) and how many tokens should be trimmed from the
          end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=1)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if tokens[-len(stop_ids) :] == stop_ids:
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def sequence_overlap(s1: Sequence, s2: Sequence) -> bool:
    """
    Checks if a suffix of s1 has overlap with a prefix of s2

    Args:
        s1 (Sequence): The first sequence
        s2 (Sequence): The second sequence

    Returns:
        bool: If the two sequences have overlap
    """
    max_overlap = min(len(s1), len(s2))
    return any(s1[-i:] == s2[:i] for i in range(1, max_overlap + 1))


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": (
            "A chat between a curious user and an artificial intelligence "
            "assistant. The assistant follows the given rules no matter what."
        ),
        "system": "ASSISTANT's RULE: ",
        "user": "USER: ",
        "assistant": "ASSISTANT: ",
        "stop": "\n",
    }
    role_mapping = role_mapping if role_mapping is not None else default_role_mapping

    prompt = ""
    for line in messages:
        role_prefix = role_mapping.get(line["role"], "")
        stop = role_mapping.get("stop", "")
        content = line.get("content", "")
        prompt += f"{role_prefix}{content}{stop}"

    prompt += role_mapping.get("assistant", "")
    return prompt.rstrip()


@dataclass
class PromptCache:
    cache: List[Any] = field(default_factory=list)
    model_key: Tuple[str, Optional[str]] = ("", None)
    tokens: List[int] = field(default_factory=list)


class ModelProvider:
    def __init__(self, cli_args: argparse.Namespace):
        """Load models on demand and persist them across the whole process."""
        self.cli_args = cli_args
        self.model_key = None
        self.model = None
        self.tokenizer = None

        # Preload the default model if it is provided
        if self.cli_args.model is not None:
            self.load("default_model")

    def _validate_model_path(self, model_path: str):
        model_path = Path(model_path)
        if model_path.exists() and not model_path.is_relative_to(Path.cwd()):
            raise RuntimeError(
                "Local models must be relative to the current working dir."
            )

    # Added in adapter_path to load dynamically
    def load(self, model_path, adapter_path=None):
        if self.model_key == (model_path, adapter_path):
            return self.model, self.tokenizer

        # Remove the old model if it exists.
        self.model = None
        self.tokenizer = None
        self.model_key = None

        # Building tokenizer_config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        if model_path == "default_model" and self.cli_args.model is not None:
            model, tokenizer = load(
                self.cli_args.model,
                adapter_path=(
                    adapter_path if adapter_path else self.cli_args.adapter_path
                ),  # if the user doesn't change the model but adds an adapter path
                tokenizer_config=tokenizer_config,
            )
        else:
            self._validate_model_path(model_path)
            model, tokenizer = load(
                model_path, adapter_path=adapter_path, tokenizer_config=tokenizer_config
            )

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = (model_path, adapter_path)
        self.model = model
        self.tokenizer = tokenizer

        return self.model, self.tokenizer


class APIHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        model_provider: ModelProvider,
        *args,
        prompt_cache: Optional[PromptCache] = None,
        system_fingerprint: Optional[str] = None,
        **kwargs,
    ):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.model_provider = model_provider
        self.prompt_cache = prompt_cache or PromptCache()
        self.system_fingerprint = system_fingerprint or get_system_fingerprint()
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _set_completion_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self._set_cors_headers()

    def _set_stream_headers(self, status_code: int = 200):
        self.send_response(status_code)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._set_cors_headers()

    def do_OPTIONS(self):
        self._set_completion_headers(204)
        self.end_headers()

    def do_POST(self):
        """
        Respond to a POST request from a client.
        """
        endpoints = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
        }

        if self.path not in endpoints:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        self.body = json.loads(raw_body.decode())
        indent = "\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Incoming Request Body: {json.dumps(self.body, indent=indent)}")
        assert isinstance(
            self.body, dict
        ), f"Request should be dict, but got {type(self.body)}"

        # Extract request parameters from the body
        self.stream = self.body.get("stream", False)
        self.stream_options = self.body.get("stream_options", None)
        self.requested_model = self.body.get("model", "default_model")
        self.adapter = self.body.get("adapters", None)
        self.max_tokens = self.body.get("max_completion_tokens", None)
        if self.max_tokens is None:
            self.max_tokens = self.body.get("max_tokens", 512)
        self.temperature = self.body.get("temperature", 1.0)
        self.top_p = self.body.get("top_p", 1.0)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.validate_model_parameters()

        # Load the model if needed
        try:
            self.model, self.tokenizer = self.model_provider.load(
                self.requested_model, self.adapter
            )
        except:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        # Get stop id sequences, if provided
        stop_words = self.body.get("stop")
        stop_words = stop_words or []
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            self.tokenizer.encode(stop_word, add_special_tokens=False)
            for stop_word in stop_words
        ]

        # Send header type
        (
            self._set_stream_headers(200)
            if self.stream
            else self._set_completion_headers(200)
        )

        # Call endpoint specific method
        prompt = endpoints[self.path]()

        # Call method based on response type
        method = self.handle_stream if self.stream else self.handle_completion
        method(prompt, stop_id_sequences)

    def validate_model_parameters(self):
        """
        Validate the model parameters passed in the request for the correct types and values.
        """
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be a boolean")

        if not isinstance(self.max_tokens, int) or self.max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")

        if not isinstance(self.temperature, (float, int)) or self.temperature < 0:
            raise ValueError("temperature must be a non-negative float")

        if not isinstance(self.top_p, (float, int)) or self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be a float between 0 and 1")

        if (
            not isinstance(self.repetition_penalty, (float, int))
            or self.repetition_penalty < 0
        ):
            raise ValueError("repetition_penalty must be a non-negative float")

        if self.logprobs != -1 and not (0 < self.logprobs <= 10):
            raise ValueError(
                f"logprobs must be between 1 and 10 but got {self.logprobs:,}"
            )

        if (
            not isinstance(self.repetition_context_size, int)
            or self.repetition_context_size < 0
        ):
            raise ValueError("repetition_context_size must be a non-negative integer")

        if self.logit_bias is not None:
            if not isinstance(self.logit_bias, dict):
                raise ValueError("logit_bias must be a dict of int to float")

            try:
                self.logit_bias = {int(k): v for k, v in self.logit_bias.items()}
            except ValueError:
                raise ValueError("logit_bias must be a dict of int to float")

        if not isinstance(self.requested_model, str):
            raise ValueError("model must be a string")
        if self.adapter is not None and not isinstance(self.adapter, str):
            raise ValueError("adapter must be a string")

    def generate_response(
        self,
        text: str,
        finish_reason: Union[Literal["length", "stop"], None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[List[float]] = None,
        top_tokens: Optional[List[Dict[int, float]]] = None,
        tokens: Optional[List[int]] = None,
    ) -> dict:
        """
        Generate a single response packet based on response type (stream or
        not), completion type and parameters.

        Args:
            text (str): Text generated by model
            finish_reason (Union[Literal["length", "stop"], None]): The reason the
              response is being sent: "length", "stop" or `None`.
            prompt_token_count (Optional[int]): The number of tokens in the prompt,
              used to populate the "usage" field (not used when stream).
            completion_token_count (Optional[int]): The number of tokens in the
              response, used to populate the "usage" field (not used when stream).
            token_logprobs (Optional[List[float]]): The log probabilities per token,
              in token order.
            top_tokens (Optional[List[Dict[int, float]]]): List of dictionaries mapping
              tokens to logprobs for the top N tokens at each token position.
            tokens (Optional[List[int]]): List of tokens to return with logprobs structure

        Returns:
            dict: A dictionary containing the response, in the same format as
              OpenAI's API.
        """
        token_logprobs = token_logprobs if token_logprobs else []
        top_logprobs = top_tokens if top_tokens else []

        # Static response
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": self.object_type,
            "model": self.requested_model,
            "created": self.created,
            "choices": [
                {
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": token_logprobs,
                        "top_logprobs": top_logprobs,
                        "tokens": tokens,
                    },
                    "finish_reason": finish_reason,
                }
            ],
        }

        if not self.stream:
            if not (
                isinstance(prompt_token_count, int)
                and isinstance(completion_token_count, int)
            ):
                raise ValueError(
                    "Response type is complete, but token counts not provided"
                )

            response["usage"] = {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            }

        choice = response["choices"][0]

        # Add dynamic response
        if self.object_type.startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            choice[key_name] = {"role": "assistant", "content": text}
        elif self.object_type == "text_completion":
            choice.update(text=text)
        else:
            ValueError(f"Unsupported response type: {self.object_type}")

        return response

    def get_prompt_cache(self, prompt):
        cache_len = len(self.prompt_cache.tokens)
        if (
            self.prompt_cache.model_key != self.model_provider.model_key
            or cache_len >= len(prompt)
            or self.prompt_cache.tokens != prompt[:cache_len]
        ):
            self.prompt_cache.model_key = self.model_provider.model_key
            self.prompt_cache.cache = make_prompt_cache(self.model_provider.model)
        else:
            prompt = prompt[cache_len:]
        self.prompt_cache.tokens.extend(prompt)
        return prompt

    def handle_completion(
        self,
        prompt: List[int],
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (List[int]): The tokenized prompt.
            stop_id_sequences (List[List[int]]): A list of stop words passed
              to the stopping_criteria function
        """
        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        tokens = []
        finish_reason = "length"
        stop_sequence_suffix = None
        logging.debug(f"Starting completion:")
        token_logprobs = []
        top_tokens = []

        prompt = self.get_prompt_cache(prompt)
        prompt = mx.array(prompt)

        for _, (token, logprobs) in zip(
            range(self.max_tokens),
            generate_step(
                prompt=prompt,
                model=self.model,
                temp=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                logit_bias=self.logit_bias,
                prompt_cache=self.prompt_cache.cache,
            ),
        ):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)

            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(tuple(top_token_info))

            token_logprobs.append(logprobs[token].item())

            stop_condition = stopping_criteria(
                tokens, stop_id_sequences, self.tokenizer.eos_token_id
            )
            if stop_condition.stop_met:
                finish_reason = "stop"
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                break

        self.prompt_cache.tokens.extend(tokens)
        detokenizer.finalize()
        text = (
            detokenizer.text
            if stop_sequence_suffix is None
            else detokenizer.text[: -len(stop_sequence_suffix)]
        )
        response = self.generate_response(
            text,
            finish_reason,
            len(prompt),
            len(tokens),
            token_logprobs=token_logprobs,
            top_tokens=top_tokens,
            tokens=tokens,
        )

        response_json = json.dumps(response).encode()
        indent = "\t"  # Backslashes can't be inside of f-strings
        logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

        # Send an additional Content-Length header when it is known
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()

        self.wfile.write(response_json)
        self.wfile.flush()

    def handle_stream(
        self,
        prompt: List[int],
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate response to prompt and foward it to the client using a Server
        Sent Events (SSE) stream.

        Args:
            prompt (mx.array): The tokenized prompt
            stop_id_sequences (List[List[int]]): A list of stop words passed to
              the stopping_criteria function
        """
        # No additional headers are needed, call end_headers
        self.end_headers()

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        tokens = []

        stop_sequence_suffix = None
        logging.debug(f"Starting stream:")

        prompt = self.get_prompt_cache(prompt)
        prompt = mx.array(prompt)

        for _, (token, _) in zip(
            range(self.max_tokens),
            generate_step(
                prompt=mx.array(prompt),
                model=self.model,
                temp=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                prompt_cache=self.prompt_cache.cache,
            ),
        ):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)

            stop_condition = stopping_criteria(
                tokens,
                stop_id_sequences,
                self.tokenizer.eos_token_id,
            )
            if stop_condition.stop_met:
                if stop_condition.trim_length:
                    stop_sequence_suffix = self.tokenizer.decode(
                        tokens[-stop_condition.trim_length :]
                    )
                break

            # If the end of tokens overlaps with a stop sequence, generate new
            # tokens until we know if the stop sequence is hit or not
            if any(
                (sequence_overlap(tokens, sequence) for sequence in stop_id_sequences)
            ):
                continue

            new_text = detokenizer.last_segment
            if new_text:
                response = self.generate_response(new_text, None)
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()

        self.prompt_cache.tokens.extend(tokens)

        # check is there any remaining text to send
        detokenizer.finalize()
        last_segment = detokenizer.last_segment
        if last_segment:
            if stop_sequence_suffix is not None:
                last_segment = last_segment[: -len(stop_sequence_suffix)]
            response = self.generate_response(last_segment, "length")
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        if self.stream_options is not None and self.stream_options["include_usage"]:
            response = self.completion_usage_response(len(prompt), len(tokens))
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())

        self.wfile.write("data: [DONE]\n\n".encode())
        self.wfile.flush()

    def completion_usage_response(
        self,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ):
        response = {
            "id": self.request_id,
            "system_fingerprint": self.system_fingerprint,
            "object": "chat.completion",
            "model": self.requested_model,
            "created": self.created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_token_count,
                "completion_tokens": completion_token_count,
                "total_tokens": prompt_token_count + completion_token_count,
            },
        }
        return response

    def handle_chat_completions(self) -> mx.array:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = (
            "chat.completions.chunk" if self.stream else "chat.completions"
        )
        if (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template
        ):
            prompt = self.tokenizer.apply_chat_template(
                body["messages"],
                body.get("tools", None),
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        return prompt

    def handle_text_completions(self) -> mx.array:
        """
        Handle a text completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"
        assert "prompt" in self.body, "Request did not contain a prompt"
        return self.tokenizer.encode(self.body["prompt"])

    def do_GET(self):
        """
        Respond to a GET request from a client.
        """
        if self.path == "/v1/models":
            self.handle_models_request()
        else:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")

    def handle_models_request(self):
        """
        Handle a GET request for the /v1/models endpoint.
        """
        self._set_completion_headers(200)
        self.end_headers()

        # Scan the cache directory for downloaded mlx models
        hf_cache_info = scan_cache_dir()
        downloaded_models = [
            repo for repo in hf_cache_info.repos if "mlx" in repo.repo_id
        ]

        # Create a list of available models
        models = [
            {
                "id": repo.repo_id,
                "object": "model",
                "created": self.created,
            }
            for repo in downloaded_models
        ]

        response = {"object": "list", "data": models}

        response_json = json.dumps(response).encode()
        self.wfile.write(response_json)
        self.wfile.flush()


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    prompt_cache = PromptCache()
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(
            model_provider,
            prompt_cache=prompt_cache,
            system_fingerprint=get_system_fingerprint(),
            *args,
            **kwargs,
        ),
    )
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    logging.info(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--cache-limit-gb",
        type=int,
        default=None,
        help="Set the MLX cache limit in GB",
        required=False,
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.cache_limit_gb is not None:
        logging.debug(f"Setting cache limit to {args.cache_limit_gb} GB")
        mx.metal.set_cache_limit(args.cache_limit_gb * 1024 * 1024 * 1024)

    run(args.host, args.port, ModelProvider(args))


if __name__ == "__main__":
    main()
