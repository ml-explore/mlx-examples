VS Code Local Coding Model with MLX
===================================

This example serves a local coding model with `mlx-lm` through an OpenAI-compatible
`/v1/chat/completions` endpoint so that VS Code can use it through `Chat: Manage Language Models`.

This targets VS Code chat, inline chat, and utility-model flows. It also exposes a
basic tool-calling path for agent-style requests by returning OpenAI-compatible
`tool_calls` responses when the model emits the expected JSON shape.

It does not replace built-in Copilot inline suggestions. Current VS Code documentation
still excludes local or BYOK models from inline code completions.

Setup
-----

Install the dependencies:

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
```

Start the server with a coding model that `mlx-lm` can load on Apple silicon:

```bash
.venv312/bin/python server.py \
  --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
```

The first run downloads the model if it is not already cached.

You can also configure the server with environment variables:

```bash
export MLX_VSCODE_MODEL=mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit
.venv312/bin/python server.py
```

By default the example runs without authentication because it is intended for a
local-only VS Code connection.

What the server exposes
-----------------------

- `GET /health` for a basic health check.
- `GET /v1/models` to advertise the local model.
- `POST /v1/chat/completions` using an OpenAI-compatible request and response shape.
- Optional function-style tool calling over the same chat completions endpoint.

Quick verification
------------------

Verify the server directly before connecting VS Code:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You write concise, correct code."},
      {"role": "user", "content": "Write a Python function that merges two sorted lists."}
    ],
    "temperature": 0.0,
    "max_tokens": 256
  }'
```

If you want to see streamed chunks, add `"stream": true` to the request body.

Hooking it into VS Code
-----------------------

Method 1 uses the VS Code UI:

1. Run `Chat: Manage Language Models` from the Command Palette.
2. Choose `Add Models`.
3. Select `Custom Endpoint`.
4. Enter any group name, for example `Local MLX`.
5. Select `Chat Completions` as the API type.
6. Point the model URL at `http://127.0.0.1:8000/v1/chat/completions`.
7. Make the model visible in the Language Models editor and select it from the chat model picker.

Method 2 uses a config file. A ready-to-edit example is included in `chatLanguageModels.example.json`.

An example provider entry looks like this:

```json
[
  {
    "name": "Local MLX",
    "vendor": "customendpoint",
    "apiType": "chat-completions",
    "models": [
      {
        "id": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        "name": "Qwen2.5 Coder 1.5B (MLX)",
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "streaming": true,
        "toolCalling": true,
        "editTools": ["find-replace", "multi-find-replace", "apply-patch", "code-rewrite"],
        "maxInputTokens": 32768,
        "maxOutputTokens": 2048
      }
    ]
  }
]
```

Suggested VS Code settings
--------------------------

You can point chat-adjacent utility tasks at the same local model:

```json
{
  "chat.utilityModel": "Local MLX/Qwen2.5 Coder 1.5B (MLX)",
  "chat.utilitySmallModel": "Local MLX/Qwen2.5 Coder 1.5B (MLX)",
  "inlineChat.defaultModel": "Local MLX/Qwen2.5 Coder 1.5B (MLX)"
}
```

The exact picker string can vary by VS Code version and provider label. Use the value that appears in your model picker.

If you want to lock the endpoint down later, the server still supports an optional
`--api-key` flag, but it is not needed for the default local setup.

Current limitations
-------------------

- Tool calling is prompt-mediated rather than model-native. Reliability depends on the coding model following the JSON contract exactly.
- Current VS Code documentation says local or BYOK models are not available for inline suggestions.
- Large coding models can require substantial RAM. Start with a quantized coding model and a low temperature.

Notes on model choice
---------------------

Good starting points for local coding assistance on Apple silicon include:

- `mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit`
- `mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit` for tighter disk and RAM budgets
- Other MLX-compatible code-tuned instruct models that `mlx-lm` can load reliably

For lower memory machines, swap in a smaller code model and keep `temperature` at `0.0` for deterministic edits.