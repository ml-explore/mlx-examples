# Fine-Tuning with Full Fine-Tuning

You can use the `mlx-lm` package to fine-tune an LLM with full fine-tuning for a target task. Full fine-tuning works with the following model families:

- Mistral
- Llama
- Phi2
- Mixtral
- Qwen2
- Gemma
- OLMo

## Contents

* [Run](#Run)
  * [Fine-tune](#Fine-tune)
  * [Evaluate](#Evaluate)
  * [Generate](#Generate)
* [Data](#Data)
* [Memory Issues](#Memory-Issues)

## Run

The main command is `mlx_lm.full`. To see a full list of command-line options run:

```shell
python -m mlx_lm.full --help
```

Note, in the following the `--model` argument can be any compatible Hugging
Face repo or a local path to a converted model.

You can also specify a YAML config with `-c`/`--config`. For more on the format see the
[example YAML](examples/lora_config.yaml). For example:

```shell
python -m mlx_lm.full --config /path/to/config.yaml
```

If command-line flags are also used, they will override the corresponding
values in the config.

### Fine-tune

To fine-tune a model use:

```shell
python -m mlx_lm.full \
    --model <path_to_model> \
    --train \
    --data <path_to_data> \
    --iters 600
```

The `--data` argument must specify a path to a `train.jsonl`, `valid.jsonl`
when using `--train` and a path to a `test.jsonl` when using `--test`. For more
details on the data format see the section on [Data](#Data).

For example, to fine-tune a Mistral 7B you can use `--model
mistralai/Mistral-7B-v0.1`.


By default, the model weights are saved in model.npz. You can specify
the output location with `--model-file`.

You can resume fine-tuning with an existing model with
`--model-file <path_to_model.npz>`.

* NOTICE:
  - After finetune, you get the model-file only,
  - You have to model only file, put it back to original model folder. (copy or else. as you wish to.)
  - You can convert model format npz to safetensor, like this.

```
# Conversion of file format
import sys
from mlx.core import *

#IN: npz format file, OUT: safetensors format file
def convert_npz_to_safetensor(input_file, output_file):
    try:
        # Load the npz file
        data = load(input_file)
        
        # Save the data as safetensor
        save_safetensors(output_file, data)
        
        print(f"Conversion successful. Output saved as {output_file}")
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except Exception as e:
        print(f"An error occurred during conversion: {str(e)}")

```

### Evaluate

To compute test set perplexity use:

```shell
python -m mlx_lm.full \
    --model-file <path_to_model.npz> \
    --data <path_to_data> \
    --test
```

### Generate

For generation use `mlx_lm.generate`:

```shell
python -m mlx_lm.generate \
    --model-file <path_to_model.npz> \
    --prompt "<your_model_prompt>"
```

## Data

The full fine-tuning command expects you to provide a dataset with `--data`.  The MLX
Examples GitHub repo has an [example of the WikiSQL
data](https://github.com/ml-explore/mlx-examples/tree/main/lora/data) in the
correct format.

For fine-tuning (`--train`), the data loader expects a `train.jsonl` and a
`valid.jsonl` to be in the data directory. For evaluation (`--test`), the data
loader expects a `test.jsonl` in the data directory. 

Currently, `*.jsonl` files support three data formats: `chat`,
`completions`, and `text`. Here are three examples of these formats:

`chat`:
  
```jsonl
{"messages": [
  {"role": "system", "content": "You are a helpful assistant." },
  {"role": "user", "content": "Hello."},
  {"role": "assistant", "content": "How can I assistant you today."},
]}
```

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

For the `chat` and `completions` formats, Hugging Face [chat
templates](https://huggingface.co/blog/chat-templates) are used. This applies
the model's chat template by default. If the model does not have a chat
template, then Hugging Face will use a default. For example, the final text in
the `chat` example above with Hugging Face's default template becomes:

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


1. Try using a smaller batch size with `--batch-size`. The default is `4` so
   setting this to `2` or `1` will reduce memory consumption. This may slow
   things down a little, but will also reduce the memory use.


2. Longer examples require more memory. If it makes sense for your data, one thing
   you can do is break your examples into smaller
   sequences when making the `{train, valid, test}.jsonl` files.

3. Gradient checkpointing lets you trade-off memory use (less) for computation
   (more) by recomputing instead of storing intermediate values needed by the
   backward pass. You can use gradient checkpointing by passing the
   `--grad-checkpoint` flag. Gradient checkpointing will be more helpful for
   larger batch sizes or sequence lengths with smaller or quantized models.

For example, Required memory is depends on model's parameter or batch size . :

```
python full.py \
    --model mistralai/Mistral-7B-v0.1 \
    --train \
    --batch-size 1 \
    --data wikisql
```

