import argparse
import json
import os
import time
import uuid
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


def is_stop_condition_met(
    tokens: List[int],
    stop_id_sequences: List[np.ndarray],
    eos_token_id: int,
    max_tokens: int,
) -> Tuple[bool, bool, int]:
    if len(tokens) >= max_tokens:
        return True, False, 0

    if tokens and tokens[-1] == eos_token_id:
        return True, True, 1

    for stop_ids in stop_id_sequences:
        if len(tokens) >= len(stop_ids):
            if np.all(np.equal(np.array(tokens[-len(stop_ids) :]), stop_ids)):
                return True, True, len(stop_ids)

    return False, False, 0


def generate(
    prompt: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    stop_id_sequences: List[np.ndarray] = None,
    eos_token_id: int = None,
    max_tokens: int = 100,
    top_p: float = 1.0,
):
    def sample(logits):
        if temp == 0:
            return mx.argmax(logits, axis=-1)
        else:
            if top_p > 0 and top_p < 1.0:
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
    tokens = []

    while True:
        logits, cache = model(y[None], cache=cache)
        logits = logits[:, -1, :]
        y = sample(logits)
        token = y.item()
        tokens.append(token)

        stop_met, trim_needed, trim_length = is_stop_condition_met(
            tokens, stop_id_sequences, eos_token_id, max_tokens
        )
        if stop_met:
            if trim_needed and trim_length > 0:
                tokens = tokens[:-trim_length]
            tokens = None
            break

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
            tokens = list(
                generate(
                    prompt,
                    _model,
                    temperature,
                    stop_id_sequences,
                    eos_token_id,
                    max_tokens,
                    top_p=top_p,
                )
            )
            text = _tokenizer.decode(tokens)
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
        else:
            self.send_response(200)
            self.send_header("Content-type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            accumulated_tokens = []
            current_generated_text_index = 0
            for token in generate(
                prompt,
                _model,
                temperature,
                stop_id_sequences,
                eos_token_id,
                max_tokens,
                top_p=top_p,
            ):
                # This is a workaround because the llama tokenizer omitted spaces during decoding token by token.
                accumulated_tokens.append(token)
                generated_text = _tokenizer.decode(accumulated_tokens)
                next_chunk = generated_text[current_generated_text_index:]
                current_generated_text_index = len(generated_text)
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
                try:
                    self.wfile.write(f"data: {json.dumps(response)}\n\n".encode())
                    self.wfile.flush()
                except Exception as e:
                    print(e)
                    break

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
