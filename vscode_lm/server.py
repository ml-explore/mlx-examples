import argparse
import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from pydantic import BaseModel, ConfigDict, Field


class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[MessagePart]]
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    type: Literal["function"]
    function: ToolFunction


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None


class ServerConfig(BaseModel):
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    api_key: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    trust_remote_code: bool = False


def build_app(config: ServerConfig) -> FastAPI:
    app = FastAPI(title="MLX VS Code Chat Endpoint")
    state: Dict[str, Any] = {"model": None, "tokenizer": None}

    def ensure_loaded() -> None:
        if state["model"] is not None:
            return

        tokenizer_config: Dict[str, Any] = {}
        if config.trust_remote_code:
            tokenizer_config["trust_remote_code"] = True

        model, tokenizer = load(config.model, tokenizer_config=tokenizer_config)
        state["model"] = model
        state["tokenizer"] = tokenizer

    def require_api_key(authorization: Optional[str]) -> None:
        if not config.api_key:
            return

        expected = f"Bearer {config.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")

    def normalize_content(content: Union[str, List[MessagePart]]) -> str:
        if isinstance(content, str):
            return content

        parts = []
        for item in content:
            if item.type == "text" and item.text:
                parts.append(item.text)
        return "".join(parts)

    def apply_stop(text: str, stop: Optional[Union[str, List[str]]]) -> str:
        if not stop:
            return text

        stops = [stop] if isinstance(stop, str) else stop
        limit = len(text)
        for marker in stops:
            idx = text.find(marker)
            if idx != -1:
                limit = min(limit, idx)
        return text[:limit]

    def count_tokens(tokenizer: Any, text: str) -> int:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            return 0

    def build_tool_instructions(
        tools: List[ToolDefinition],
        tool_choice: Optional[Union[str, Dict[str, Any]]],
    ) -> str:
        tool_specs = []
        for tool in tools:
            tool_specs.append(
                {
                    "name": tool.function.name,
                    "description": tool.function.description or "",
                    "parameters": tool.function.parameters,
                }
            )

        choice_note = "Choose the best tool only when it is necessary."
        if isinstance(tool_choice, str):
            if tool_choice == "required":
                choice_note = "You must call exactly one tool before producing any final answer."
            elif tool_choice == "none":
                choice_note = "Do not call tools. Answer directly."
        elif isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name:
                choice_note = f"You must call the tool named {name}."

        return (
            "You can call tools. "
            + choice_note
            + " When calling a tool, reply with only valid JSON using this exact schema: "
            + '{"tool_calls":[{"name":"tool_name","arguments":{}}]}'
            + ". Do not wrap the JSON in markdown fences. "
            + "If no tool is needed, answer normally in plain text. "
            + "Available tools: "
            + json.dumps(tool_specs, ensure_ascii=True)
        )

    def build_prompt(
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> str:
        tokenizer = state["tokenizer"]
        formatted_messages = []
        if tools:
            formatted_messages.append(
                {
                    "role": "system",
                    "content": build_tool_instructions(tools, tool_choice),
                }
            )

        for message in messages:
            content = normalize_content(message.content)
            if message.role == "tool":
                label = message.name or message.tool_call_id or "tool"
                formatted_messages.append(
                    {
                        "role": "user",
                        "content": f"Tool result from {label}:\n{content}",
                    }
                )
                continue

            formatted_messages.append({"role": message.role, "content": content})

        return tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def parse_tool_calls(
        text: str,
        tools: Optional[List[ToolDefinition]],
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        raw_calls = payload.get("tool_calls")
        if not isinstance(raw_calls, list) or not raw_calls:
            return None

        allowed_tools = {tool.function.name for tool in tools}
        parsed_calls = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                return None

            name = raw_call.get("name")
            arguments = raw_call.get("arguments", {})
            if name not in allowed_tools:
                return None
            if not isinstance(arguments, dict):
                return None

            parsed_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments, ensure_ascii=True),
                    },
                }
            )

        return parsed_calls

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models(
        authorization: Optional[str] = Header(default=None),
    ) -> Dict[str, Any]:
        require_api_key(authorization)
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model,
                    "object": "model",
                    "owned_by": "local-mlx",
                }
            ],
        }

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        request: ChatCompletionRequest,
        authorization: Optional[str] = Header(default=None),
    ) -> Union[JSONResponse, StreamingResponse]:
        require_api_key(authorization)
        ensure_loaded()

        model = state["model"]
        tokenizer = state["tokenizer"]
        prompt = build_prompt(request.messages, request.tools, request.tool_choice)
        max_tokens = request.max_tokens or config.max_tokens
        temperature = (
            config.temperature if request.temperature is None else request.temperature
        )
        sampler = make_sampler(temp=temperature)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if request.stream:
            async def event_stream() -> AsyncIterator[str]:
                yield "data: " + json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model or config.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                ) + "\n\n"

                full_text = ""
                for response in stream_generate(
                    model,
                    tokenizer,
                    prompt,
                    max_tokens=max_tokens,
                    sampler=sampler,
                ):
                    full_text += response.text
                    if apply_stop(full_text, request.stop) != full_text:
                        break

                completion_text = apply_stop(full_text, request.stop)
                prompt_tokens = count_tokens(tokenizer, prompt)
                completion_tokens = count_tokens(tokenizer, completion_text)
                tool_calls = parse_tool_calls(completion_text, request.tools)

                if tool_calls:
                    yield "data: " + json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model or config.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": tool_calls},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ) + "\n\n"
                else:
                    yield "data: " + json.dumps(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": request.model or config.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": completion_text},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ) + "\n\n"

                yield "data: " + json.dumps(
                    {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": request.model or config.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if tool_calls else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                ) + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        full_text = ""
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            full_text += response.text
            if apply_stop(full_text, request.stop) != full_text:
                break

        text = apply_stop(full_text, request.stop)
        prompt_tokens = count_tokens(tokenizer, prompt)
        completion_tokens = count_tokens(tokenizer, text)
        tool_calls = parse_tool_calls(text, request.tools)
        message: Dict[str, Any] = {"role": "assistant", "content": text}
        finish_reason = "stop"
        if tool_calls:
            message["content"] = None
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"

        payload = {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model or config.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return JSONResponse(payload)

    return app


def parse_args() -> ServerConfig:
    parser = argparse.ArgumentParser(
        description="Serve a local mlx-lm model through an OpenAI-compatible chat API."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MLX_VSCODE_MODEL", "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit"),
        help="MLX-compatible model identifier or local path.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("MLX_VSCODE_HOST", "127.0.0.1"),
        help="Bind host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MLX_VSCODE_PORT", "8000")),
        help="Bind port.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MLX_VSCODE_API_KEY"),
        help="Optional bearer token required by the server.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("MLX_VSCODE_MAX_TOKENS", "512")),
        help="Default completion budget per request.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("MLX_VSCODE_TEMPERATURE", "0.0")),
        help="Default sampling temperature.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to the tokenizer configuration.",
    )
    args = parser.parse_args()
    return ServerConfig(**vars(args))


if __name__ == "__main__":
    import uvicorn

    config = parse_args()
    uvicorn.run(build_app(config), host=config.host, port=config.port)