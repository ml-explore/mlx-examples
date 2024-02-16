import argparse
import json
import time
import uuid
from collections import namedtuple
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import PreTrainedTokenizer

from .utils import load

_model: Optional[nn.Module] = None
_tokenizer: Optional[PreTrainedTokenizer] = None


def load_model(model_path: str, adapter_file: Optional[str] = None):
    global _model
    global _tokenizer
    _model, _tokenizer = load(model_path, adapter_file=adapter_file)


StopCondition = namedtuple("StopCondition", ["stop_met", "trim_length"])


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
        return StopCondition(stop_met=True, trim_length=0)

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if np.array_equal(tokens[-len(stop_ids) :], stop_ids):
                return StopCondition(stop_met=True, trim_length=len(stop_ids))

    return StopCondition(stop_met=False, trim_length=0)


def generate(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    top_p: float = 1.0,
):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
                if (
                    logits.dtype == mx.bfloat16
                ):  # workdaround for unable to load kernel contiguous_scan_inclusive_sum_bfloat16_bfloat16
                    logits = logits.astype(mx.float32)
                probs = mx.softmax(logits / temp, axis=-1)

                sorted_probs = mx.sort(probs)[::-1]
                sorted_indices = mx.argsort(probs)[::-1]
                cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

                top_probs = mx.where(
                    cumulative_probs > 1 - top_p,
                    sorted_probs,
                    mx.zeros_like(sorted_probs),
                )
                sorted_tok = mx.random.categorical(mx.log(top_probs))
                tok = sorted_indices.squeeze(0)[sorted_tok]
                return tok
        return mx.random.categorical(logits * (1 / temp))

    y = prompt
    cache = None

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]

        y = sample(logits)
        token = y.item()

        yield token


def convert_chat(messages: any, role_mapping: Optional[dict] = None):
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


def create_response(chat_id, requested_model, prompt, tokens, text):
    response = {
        "id": chat_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "system_fingerprint": f"fp_{uuid.uuid4()}",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "logprobs": None,
                "finish_reason": None,
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt),
            "completion_tokens": len(tokens),
            "total_tokens": len(prompt) + len(tokens),
        },
    }

    return response


def create_chunk_response(chat_id, requested_model, next_chunk):
    response = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": requested_model,
        "system_fingerprint": f"fp_{uuid.uuid4()}",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": next_chunk},
                "logprobs": None,
                "finish_reason": None,
            }
        ],
    }
    return response


class APIHandler(BaseHTTPRequestHandler):
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
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            self._set_headers(200)

            response = self.handle_post_request(post_data)

            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self._set_headers(404)
            self.wfile.write(b"Not Found")

    def handle_post_request(self, post_data):
        body = json.loads(post_data.decode("utf-8"))
        chat_id = f"chatcmpl-{uuid.uuid4()}"
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
            _tokenizer.encode(stop_word, return_tensors="np", add_special_tokens=False)[
                0
            ]
            for stop_word in stop_words
        ]
        eos_token_id = _tokenizer.eos_token_id
        max_tokens = body.get("max_tokens", 100)
        stream = body.get("stream", False)
        requested_model = body.get("model", "default_model")
        temperature = body.get("temperature", 1.0)
        top_p = body.get("top_p", 1.0)
        if not stream:
            tokens = []
            for token, _ in zip(
                generate(
                    prompt,
                    _model,
                    temperature,
                    top_p=top_p,
                ),
                range(max_tokens),
            ):
                tokens.append(token)
                stop_condition = stopping_criteria(
                    tokens, stop_id_sequences, eos_token_id
                )
                if stop_condition.stop_met:
                    if stop_condition.trim_length:
                        tokens = tokens[: -stop_condition.trim_length]
                    break

            text = _tokenizer.decode(tokens)
            return create_response(chat_id, requested_model, prompt, tokens, text)
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            max_stop_id_sequence_len = (
                max(len(seq) for seq in stop_id_sequences) if stop_id_sequences else 0
            )
            tokens = []
            current_generated_text_index = 0
            # Buffer to store the last `max_stop_id_sequence_len` tokens to check for stop conditions before writing to the stream.
            stop_sequence_buffer = []
            REPLACEMENT_CHAR = "\ufffd"
            for token, _ in zip(
                generate(
                    prompt,
                    _model,
                    temperature,
                    top_p=top_p,
                ),
                range(max_tokens),
            ):
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

                    response = create_chunk_response(
                        chat_id, requested_model, next_chunk
                    )
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
                response = create_chunk_response(chat_id, requested_model, next_chunk)
                try:
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                except Exception as e:
                    print(e)

            self.wfile.write(f"data: [DONE]\n\n".encode())
            self.wfile.flush()


def run(host: str, port: int, server_class=HTTPServer, handler_class=APIHandler):
    server_address = (host, port)
    httpd = server_class(server_address, handler_class)
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
