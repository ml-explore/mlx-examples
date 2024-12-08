# Fine-Tuning with LoRA or QLoRA

You can use use the `mlx-lm` package to fine-tune an LLM with low rank
adaptation (LoRA) for a target task.[^lora] The example also supports quantized
LoRA (QLoRA).[^qlora] LoRA fine-tuning works with the following model families:

- Mistral
- Llama
- Phi2
- Mixtral
- Qwen2
- Gemma
- OLMo
- MiniCPM
- InternLM2

## Contents

- [Run](#Run)
  - [Fine-tune](#Fine-tune)
  - [Evaluate](#Evaluate)
  - [Generate](#Generate)
- [Fuse](#Fuse)
- [Data](#Data)
- [Memory Issues](#Memory-Issues)

## Run

The main command is `mlx_lm.lora`. To see a full list of command-line options run:

```shell
mlx_lm.lora --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted model.

You can also specify a YAML config with `-c`/`--config`. For more on the format see the
[example YAML](examples/lora_config.yaml). For example:

```shell
mlx_lm.lora --config /path/to/config.yaml
```

If command-line flags are also used, they will override the corresponding
values in the config.

### Fine-tune

To fine-tune a model use:

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --iters 600
```

To fine-tune the full model weights, add the `--fine-tune-type full` flag.
Currently supported fine-tuning types are `lora` (default), `dora`, and `full`.

The `--data` argument must specify a path to a `train.jsonl`, `valid.jsonl`
when using `--train` and a path to a `test.jsonl` when using `--test`. For more
details on the data format see the section on [Data](#Data).

For example, to fine-tune a Mistral 7B you can use `--model
mistralai/Mistral-7B-v0.1`.

If `--model` points to a quantized model, then the training will use QLoRA,
otherwise it will use regular LoRA.

By default, the adapter config and learned weights are saved in `adapters/`.
You can specify the output location with `--adapter-path`.

You can resume fine-tuning with an existing adapter with
`--resume-adapter-file <path_to_adapters.safetensors>`.

### Input Masking
There are custom functions for masking the sequence of tokens associated with the `prompt` in a completion dataset
during the loss calculation to ensure the model is not being penalized for not recreating the prompt.  To fine-tune 
with masked input sequences, use the `--mask-inputs` argument.

This functionality expects a ```response_template``` parameter in the configuration that is either a string representing
a [string that indicate the start of the model's response](https://huggingface.co/docs/transformers/en/chat_templating#what-are-generation-prompts) 
or its corresopnding tokens.  This is used to create the mask that excludes the tokens associated from the rest of
the sequence from loss calculations.  For example (ChatML):

```yaml
response_template: "<|im_start|>assistant"
```

or (for the corresponding tokens of Gemma's response template)

```yaml
response_template: [106, 2516]
```


### Evaluate

To compute test set perplexity use:

```shell
mlx_lm.lora \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --data <path_to_data> \
    --test
```

### Generate

For generation use `mlx_lm.generate`:

```shell
mlx_lm.generate \
    --model <path_to_model> \
    --adapter-path <path_to_adapters> \
    --prompt "<your_model_prompt>"
```

## Fuse

You can generate a model fused with the low-rank adapters using the
`mlx_lm.fuse` command. This command also allows you to optionally:

- Upload the fused model to the Hugging Face Hub.
- Export the fused model to GGUF. Note GGUF support is limited to Mistral,
  Mixtral, and Llama style models in fp16 precision.

To see supported options run:

```shell
mlx_lm.fuse --help
```

To generate the fused model run:

```shell
mlx_lm.fuse --model <path_to_model>
```

This will by default load the adapters from `adapters/`, and save the fused
model in the path `fused_model/`. All of these are configurable.

To upload a fused model, supply the `--upload-repo` and `--hf-path` arguments
to `mlx_lm.fuse`. The latter is the repo name of the original model, which is
useful for the sake of attribution and model versioning.

For example, to fuse and upload a model derived from Mistral-7B-v0.1, run:

```shell
mlx_lm.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --upload-repo mlx-community/my-lora-mistral-7b \
    --hf-path mistralai/Mistral-7B-v0.1
```

To export a fused model to GGUF, run:

```shell
mlx_lm.fuse \
    --model mistralai/Mistral-7B-v0.1 \
    --export-gguf
```

This will save the GGUF model in `fused_model/ggml-model-f16.gguf`. You
can specify the file name with `--gguf-path`.

## Data

The LoRA command expects you to provide a dataset with `--data`. The MLX
Examples GitHub repo has an [example of the WikiSQL
data](https://github.com/ml-explore/mlx-examples/tree/main/lora/data) in the
correct format.

Datasets can be specified in `*.jsonl` files locally or loaded from Hugging
Face. 

### Local Datasets

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a
`valid.jsonl` to be in the data directory. For evaluation (`--test`), the data
loader expects a `test.jsonl` in the data directory. 

Currently, `*.jsonl` files support `chat`, `tools`, `completions`, and `text`
data formats. Here are examples of these formats:

`chat`:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello."}, {"role": "assistant", "content": "How can I assistant you today."}]}
```

`tools`:

```jsonl
{"messages":[{"role":"user","content":"What is the weather in San Francisco?"},{"role":"assistant","tool_calls":[{"id":"call_id","type":"function","function":{"name":"get_current_weather","arguments":"{\"location\": \"San Francisco, USA\", \"format\": \"celsius\"}"}}]}],"tools":[{"type":"function","function":{"name":"get_current_weather","description":"Get the current weather","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city and country, eg. San Francisco, USA"},"format":{"type":"string","enum":["celsius","fahrenheit"]}},"required":["location","format"]}}}]}
```

<details>
<summary>View the expanded single data tool format</summary>

```jsonl
{
    "messages": [
        { "role": "user", "content": "What is the weather in San Francisco?" },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_id",
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "arguments": "{\"location\": \"San Francisco, USA\", \"format\": \"celsius\"}"
                    }
                }
            ]
        }
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and country, eg. San Francisco, USA"
                        },
                        "format": { "type": "string", "enum": ["celsius", "fahrenheit"] }
                    },
                    "required": ["location", "format"]
                }
            }
        }
    ]
}
```


The format for the `arguments` field in a function varies for different models.
Common formats include JSON strings and dictionaries. The example provided
follows the format used by
[OpenAI](https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-examples)
and [Mistral
AI](https://github.com/mistralai/mistral-finetune?tab=readme-ov-file#instruct).
A dictionary format is used in Hugging Face's [chat
templates](https://huggingface.co/docs/transformers/main/en/chat_templating#a-complete-tool-use-example).
Refer to the documentation for the model you are fine-tuning for more details.

</details>

`completions`:

```jsonl
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

`text`:

```jsonl
{"text": "This is an example for the model."}
```

Note, the format is automatically determined by the dataset. Note also, keys in
each line not expected by the loader will be ignored.

> [!NOTE]
> Each example in the datasets must be on a single line. Do not put more than
> one example per line and do not split an example across multiple lines.

### Hugging Face Datasets

To use Hugging Face datasets, first install the `datasets` package:

```
pip install datasets
```

If the Hugging Face dataset is already in a supported format, you can specify
it on the command line. For example, pass `--data mlx-community/wikisql` to
train on the pre-formatted WikiwSQL data.

Otherwise, provide a mapping of keys in the dataset to the features MLX LM
expects. Use a YAML config to specify the Hugging Face (HF)  dataset arguments. For
example:

```
hf_dataset:
  name: "billsum"
  prompt_feature: "text"
  completion_feature: "summary"
```

- Use `prompt_feature` and `completion_feature` to specify keys for a
  `completions` dataset. Use `text_feature` to specify the key for a `text`
  dataset. Use `chat_feature` to specify the key for a chat dataset.

- To specify the train, valid, or test splits, set the corresponding
  `{train,valid,test}_split` argument. 

You can specify a list of HF datasets using the `hf_datasets` (plural) configuration, which is a list of records
each with the same structure as above.  For example:

```yaml
hf_datasets: 
- hf_dataset:
    name: "Open-Orca/OpenOrca"
    train_split: "train[:90%]"
    valid_split: "train[-10%:]"
    prompt_feature: "question"
    completion_feature: "response"
- hf_dataset:
    name: "trl-lib/ultrafeedback_binarized"
    train_split: "train[:90%]"
    valid_split: "train[-10%:]"
    chat_feature: "chosen"
```

- Arguments specified in `config` will be passed as keyword arguments to
  [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/loading_methods#datasets.load_dataset).

In general, for the `chat`, `tools` and `completions` formats, Hugging Face
[chat
templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
are used. This applies the model's chat template by default. If the model does
not have a chat template, then Hugging Face will use a default. For example,
the final text in the `chat` example above with Hugging Face's default template
becomes:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello.<|im_end|>
<|im_start|>assistant
How can I assistant you today.<|im_end|>
```

If you are unsure of the format to use, the `chat` or `completions` are good to
start with. For custom requirements on the format of the dataset, use the
`text` format to assemble the content yourself.

## Memory Issues

Fine-tuning a large model with LoRA requires a machine with a decent amount
of memory. Here are some tips to reduce memory use should you need to do so:

1. Try quantization (QLoRA). You can use QLoRA by generating a quantized model
   with `convert.py` and the `-q` flag. See the [Setup](#setup) section for
   more details.

2. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.

3. Reduce the number of layers to fine-tune with `--num-layers`. The default
   is `16`, so you can try `8` or `4`. This reduces the amount of memory
   needed for back propagation. It may also reduce the quality of the
   fine-tuned model if you are fine-tuning with a lot of data.

4. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

5. Gradient checkpointing lets you trade-off memory use (less) for computation
   (more) by recomputing instead of storing intermediate values needed by the
   backward pass. You can use gradient checkpointing by passing the
   `--grad-checkpoint` flag. Gradient checkpointing will be more helpful for
   larger batch sizes or sequence lengths with smaller or quantized models.

For example, for a machine with 32 GB the following should run reasonably fast:

```
mlx_lm.lora \
    --model mistralai/Mistral-7B-v0.1 \
    --train \
    --batch-size 1 \
    --num-layers 4 \
    --data wikisql
```

The above command on an M1 Max with 32 GB runs at about 250
tokens-per-second, using the MLX Example
[`wikisql`](https://github.com/ml-explore/mlx-examples/tree/main/lora/data)
data set.

[^lora]: Refer to the [arXiv paper](https://arxiv.org/abs/2106.09685) for more details on LoRA.

[^qlora]: Refer to the paper [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
