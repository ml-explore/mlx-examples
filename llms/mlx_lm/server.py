# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import logging
import time
import uuid
import warnings
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Literal, NamedTuple, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from transformers import PreTrainedTokenizer

from .tokenizer_utils import TokenizerWrapper
from .utils import generate_step, load


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

    def load(self, model_path):
        if self.model_key == model_path:
            return self.model, self.tokenizer

        # Remove the old model if it exists.
        self.model = None
        self.tokenizer = None

        # Building tokenizer_config
        tokenizer_config = {
            "trust_remote_code": True if self.cli_args.trust_remote_code else None
        }
        if self.cli_args.chat_template:
            tokenizer_config["chat_template"] = self.cli_args.chat_template

        if model_path == "default_model" and self.cli_args.model is not None:
            model, tokenizer = load(
                self.cli_args.model,
                adapter_path=self.cli_args.adapter_path,
                tokenizer_config=tokenizer_config,
            )
        else:
            self._validate_model_path(model_path)
            model, tokenizer = load(model_path, tokenizer_config=tokenizer_config)

        if self.cli_args.use_default_chat_template:
            if tokenizer.chat_template is None:
                tokenizer.chat_template = tokenizer.default_chat_template

        self.model_key = model_path
        self.model = model
        self.tokenizer = tokenizer

        return self.model, self.tokenizer


class APIHandler(BaseHTTPRequestHandler):
    def __init__(self, model_provider: ModelProvider, *args, **kwargs):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())
        self.model_provider = model_provider
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
        self.requested_model = self.body.get("model", "default_model")
        self.max_tokens = self.body.get("max_tokens", 100)
        self.temperature = self.body.get("temperature", 1.0)
        self.top_p = self.body.get("top_p", 1.0)
        self.repetition_penalty = self.body.get("repetition_penalty", 1.0)
        self.repetition_context_size = self.body.get("repetition_context_size", 20)
        self.logit_bias = self.body.get("logit_bias", None)
        self.logprobs = self.body.get("logprobs", -1)
        self.validate_model_parameters()

        # Load the model if needed
        try:
            self.model, self.tokenizer = self.model_provider.load(self.requested_model)
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
            "system_fingerprint": f"fp_{uuid.uuid4()}",
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

    def handle_completion(
        self,
        prompt: mx.array,
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate a response to a prompt and send it to the client in a single batch.

        Args:
            prompt (mx.array): The prompt, in token form inside of a mlx array
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
        for (token, logprobs), _ in zip(
            generate_step(
                prompt=prompt,
                model=self.model,
                temp=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
                logit_bias=self.logit_bias,
            ),
            range(self.max_tokens),
        ):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)

            if self.logprobs > 0:
                sorted_indices = mx.argpartition(-logprobs, kth=self.logprobs - 1)
                top_indices = sorted_indices[: self.logprobs]
                top_logprobs = logprobs[top_indices]
                top_token_info = zip(top_indices.tolist(), top_logprobs.tolist())
                top_tokens.append(dict(top_token_info))

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
        prompt: mx.array,
        stop_id_sequences: List[List[int]],
    ):
        """
        Generate response to prompt and foward it to the client using a Server Sent Events (SSE) stream.

        Args:
            prompt (mx.array): The prompt, in token form inside of a mlx array
            stop_id_sequences (List[List[int]]):
                A list of stop words passed to the stopping_criteria function
        """
        # No additional headers are needed, call end_headers
        self.end_headers()

        detokenizer = self.tokenizer.detokenizer
        detokenizer.reset()
        tokens = []

        max_stop_id_sequence_len = len(max(stop_id_sequences, default=[]))
        # Buffer to store the last `max_stop_id_sequence_len` tokens
        # to check for stop conditions before writing to the stream.
        stop_sequence_buffer = []
        stop_sequence_suffix = None
        logging.debug(f"Starting stream:")
        for (token, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=self.model,
                temp=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            ),
            range(self.max_tokens),
        ):
            detokenizer.add_token(token)
            logging.debug(detokenizer.text)
            tokens.append(token)
            stop_sequence_buffer.append(token)

            # Continue generating tokens until buffer is as large as the longest stop_id_sequence
            if len(stop_sequence_buffer) < max_stop_id_sequence_len:
                continue

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

            new_text = detokenizer.last_segment
            response = self.generate_response(new_text, None)
            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()
            stop_sequence_buffer = []

        # check is there any remaining text to send
        if stop_sequence_buffer:
            next_chunk = (
                detokenizer.last_segment
                if stop_sequence_suffix is None
                else detokenizer.last_segment[: -len(stop_sequence_suffix)]
            )
            response = self.generate_response(next_chunk, "length")

            self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
            self.wfile.flush()

        self.wfile.write("data: [DONE]\n\n".encode())
        self.wfile.flush()

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
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)

        return mx.array(prompt)

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
        prompt_text = self.body["prompt"]
        prompt = self.tokenizer.encode(prompt_text)
        return mx.array(prompt)


def run(
    host: str,
    port: int,
    model_provider: ModelProvider,
    server_class=HTTPServer,
    handler_class=APIHandler,
):
    server_address = (host, port)
    httpd = server_class(
        server_address,
        lambda *args, **kwargs: handler_class(model_provider, *args, **kwargs),
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
