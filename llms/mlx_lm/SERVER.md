# HTTP Model Server

You use `mlx-lm` to make an HTTP API for generating text with any supported
model. The HTTP API is intended to be similar to the [OpenAI chat
API](https://platform.openai.com/docs/api-reference).

> [!NOTE]  
> The MLX LM server is not recommended for production as it only implements
> basic security checks.

Start the server with: 

```shell
python -m mlx_lm.server --model <path_to_model_or_hf_repo>
```

For example:

```shell
python -m mlx_lm.server --model mistralai/Mistral-7B-Instruct-v0.1
```

This will start a text generation server on port `8080` of the `localhost`
using Mistral 7B instruct. The model will be downloaded from the provided
Hugging Face repo if it is not already in the local cache.

To see a full list of options run:

```shell
python -m mlx_lm.server --help
```

You can make a request to the model by running:

```shell
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

### Request Fields

- `messages`: An array of message objects representing the conversation
  history. Each message object should have a role (e.g. user, assistant) and
  content (the message text).

- `role_mapping`: (Optional) A dictionary to customize the role prefixes in
  the generated prompt. If not provided, the default mappings are used.

- `stop`: (Optional) An array of strings or a single string. Thesse are
  sequences of tokens on which the generation should stop.

- `max_tokens`: (Optional) An integer specifying the maximum number of tokens
  to generate. Defaults to `100`.

- `stream`: (Optional) A boolean indicating if the response should be
  streamed. If true, responses are sent as they are generated. Defaults to
  false.

- `temperature`: (Optional) A float specifying the sampling temperature.
  Defaults to `1.0`.

- `top_p`: (Optional) A float specifying the nucleus sampling parameter.
  Defaults to `1.0`.

- `repetition_penalty`: (Optional) Applies a penalty to repeated tokens.
  Defaults to `1.0`.

- `repetition_context_size`: (Optional) The size of the context window for
  applying repetition penalty. Defaults to `20`.
