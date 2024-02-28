# Copyright © 2023-2024 Apple Inc.

import argparse
import json
import time
import uuid
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, List, NamedTuple, Optional, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import PreTrainedTokenizer

from .utils import generate_step, load

_model: Optional[nn.Module] = None
_tokenizer: Optional[PreTrainedTokenizer] = None


def load_model(model_path: str, adapter_file: Optional[str] = None):
    global _model
    global _tokenizer
    _model, _tokenizer = load(model_path, adapter_file=adapter_file)


class StopCondition(NamedTuple):
    stop_met: bool
    trim_length: int


def stopping_criteria(
    tokens: List[int],
    stop_id_sequences: List[np.ndarray],
    eos_token_id: int,
) -> StopCondition:
    """
    Determines whether the token generation should stop based on predefined conditions.

    Args:
        tokens (List[int]): The current sequence of generated tokens.
        stop_id_sequences (List[np.ndarray]): A list of numpy arrays, each representing a sequence of token IDs.
            If the end of the `tokens` list matches any of these sequences, the generation should stop.
        eos_token_id (int): The token ID that represents the end-of-sequence. If the last token in `tokens` matches this,
            the generation should stop.

    Returns:
        StopCondition: A named tuple indicating whether the stop condition has been met (`stop_met`)
            and how many tokens should be trimmed from the end if it has (`trim_length`).
    """
    if tokens and tokens[-1] == eos_token_id:
        return StopCondition(stop_met=True, trim_length=1)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if np.array_equal(tokens[-len(stop_ids) :], stop_ids):
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def convert_chat(messages: List[dict], role_mapping: Optional[dict] = None):
    default_role_mapping = {
        "system_prompt": "A chat between a curious user and an artificial intelligence assistant. The assistant follows the given rules no matter what.",
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


class APIHandler(BaseHTTPRequestHandler):

    # System fingerprint is the same for all instances of the class
    system_fingerprint = f"fp_{uuid.uuid4()}"

    def __init__(self):
        """
        Create static request specific metadata
        """
        self.created = int(time.time())

    def _set_headers(self, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_POST(self):
        endpoints = {
            "/v1/completions": self.handle_completions,
            "/v1/chat/completions": self.handle_chat_completions,
        }

        if self.path not in endpoints:
            self._set_headers(404)
            self.wfile.write(b"Not Found")
            return

        # Fetch and parse request body
        content_length = int(self.headers["Content-Length"])
        raw_body = self.rfile.read(content_length)
        self.body = json.loads(raw_body.decode())
        assert isinstance(self.body, dict), f"Request should be dict, but got {type(self.body)}"

        self.stream = self.body.get("stream", False)
        self.requested_model = self.body.get("model", "default_model")

        self._set_headers(200)
        endpoints[self.path]()

    def generate_response(
        self,
        text: str,
        finish_reason: Union[str, None],
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
    ) -> dict:

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
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ]
        }

        if self.stream:
            if not (isinstance(prompt_token_count, int) and isinstance(completion_token_count, int)):
                raise ValueError("Response type is stream, but token counts not provided")

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

    def generate_completion(
        self,
        prompt: mx.array,
        response_id: str,
        requested_model: str,
        stop_id_sequences: List[np.ndarray],
        eos_token_id: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: Optional[float],
        repetition_context_size: Optional[int],
        response_creator: Callable[[str, str, mx.array, list[int], str]],
    ):
        tokens = []
        for (token, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=_model,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            ),
            range(max_tokens),
        ):
            token = token.item()
            tokens.append(token)
            stop_condition = stopping_criteria(tokens, stop_id_sequences, eos_token_id)
            if stop_condition.stop_met:
                if stop_condition.trim_length:
                    tokens = tokens[: -stop_condition.trim_length]
                break

        text = _tokenizer.decode(tokens)
        response = self.generate_response(text, "stop", len(prompt), len(tokens))
        try:
            self.wfile.write(f"{response}\n\n".encode())
            self.wfile.flush()
        except Exception as error:
            print(error)

    def handle_stream(
        self,
        prompt: mx.array,
        response_id: str,
        requested_model: str,
        stop_id_sequences: List[np.ndarray],
        eos_token_id: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: Optional[float],
        repetition_context_size: Optional[int],
        response_creator: Callable[[str, str, str]],
    ):
        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        max_stop_id_sequence_len = max(stop_id_sequences, key=len) if stop_id_sequences else 0
        tokens = []
        current_generated_text_index = 0
        # Buffer to store the last `max_stop_id_sequence_len` tokens to check for stop conditions before writing to the stream.
        stop_sequence_buffer = []
        REPLACEMENT_CHAR = "\ufffd"
        for (token, _), _ in zip(
            generate_step(
                prompt=prompt,
                model=_model,
                temp=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            ),
            range(max_tokens),
        ):
            token = token.item()
            tokens.append(token)
            stop_sequence_buffer.append(token)
            if len(stop_sequence_buffer) > max_stop_id_sequence_len:
                if REPLACEMENT_CHAR in _tokenizer.decode(token):
                    continue
                stop_condition = stopping_criteria(
                    tokens,
                    stop_id_sequences,
                    eos_token_id,
                )
                if stop_condition.stop_met:
                    if stop_condition.trim_length:
                        tokens = tokens[: -stop_condition.trim_length]
                    break
                # This is a workaround because the llama tokenizer emits spaces when decoding token by token.
                generated_text = _tokenizer.decode(tokens)
                next_chunk = generated_text[current_generated_text_index:]
                current_generated_text_index = len(generated_text)

                response = self.generate_response(next_chunk, None)
                response = response_creator(response_id, requested_model, next_chunk)
                try:
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                    stop_sequence_buffer = []
                except Exception as e:
                    print(e)
                    break

        # check is there any remaining text to send
        if stop_sequence_buffer:
            generated_text = _tokenizer.decode(tokens)
            next_chunk = generated_text[current_generated_text_index:]
            response = self.generate_response(next_chunk, "length")
            try:
                self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                self.wfile.flush()
            except Exception as e:
                print(e)

        self.wfile.write(f"data: [DONE]\n\n".encode())
        self.wfile.flush()

    def handle_chat_completions(self):

        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completions.chunk" if self.stream else "chat.completions"

        if hasattr(_tokenizer, "apply_chat_template") and _tokenizer.chat_template:
            prompt = _tokenizer.apply_chat_template(
                body["messages"],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="np",
            )
        else:
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = _tokenizer.encode(prompt, return_tensors="np")

        prompt = mx.array(prompt[0])
        stop_words = body.get("stop", [])
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            _tokenizer.encode(stop_word, return_tensors="np", add_special_tokens=False)[0]
            for stop_word in stop_words
        ]
        eos_token_id = _tokenizer.eos_token_id
        max_tokens = body.get("max_tokens", 100)
        self.requested_model = body.get("model", "default_model")
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        repetition_penalty = body.get("repetition_penalty", 1.0)
        repetition_context_size = body.get("repetition_context_size", 20)

        method = self.handle_stream if self.stream else self.generate_completion
        method(
            prompt,
            chat_id,
            requested_model,
            stop_id_sequences,
            eos_token_id,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            repetition_context_size,
            create_chat_response,
        )

    def handle_completions(self):

        body = self.body
        assert "prompt" in body, "Request did not contain a prompt"

        # Determine response type
        self.request_id = f"cmpl-{uuid.uuid4()}"
        self.object_type = "text_completion"

        prompt_text = body["prompt"]

        prompt = _tokenizer.encode(prompt_text, return_tensors="np")
        prompt = mx.array(prompt[0])
        stop_words = body.get("stop", [])
        stop_words = [stop_words] if isinstance(stop_words, str) else stop_words
        stop_id_sequences = [
            _tokenizer.encode(stop_word, return_tensors="np", add_special_tokens=False)[0]
            for stop_word in stop_words
        ]
        eos_token_id = _tokenizer.eos_token_id
        max_tokens = body.get("max_tokens", 100)
        requested_model = body.get("model", "default_model")
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        repetition_penalty = body.get("repetition_penalty", 1.0)
        repetition_context_size = body.get("repetition_context_size", 20)

        method = self.handle_stream if self.stream else self.generate_completion
        method(
            prompt,
            completion_id,
            requested_model,
            stop_id_sequences,
            eos_token_id,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            repetition_context_size,
            create_completion_response,
        )


def run(host: str, port: int, server_class=HTTPServer, handler_class=APIHandler):
    server_address = (host, port)
    httpd = server_class(server_address, handler_class)
    warnings.warn(
        "mlx_lm.server is not recommended for production as "
        "it only implements basic security checks."
    )
    print(f"Starting httpd at {host} on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-file",
        type=str,
        help="Optional path for the trained adapter weights.",
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
    args = parser.parse_args()

    load_model(args.model, adapter_file=args.adapter_file)

    run(args.host, args.port)
