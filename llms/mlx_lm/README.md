## Generate Text with MLX and :hugs: Hugging Face

This an example of large language model text generation that can pull models from
the Hugging Face Hub.

For more information on this example, see the [README](../README.md) in the
parent directory.

This package also supports fine tuning with LoRA or QLoRA. For more information
see the [LoRA documentation](LORA.md).


## Install mlx_lm locally

```shell 
# go to the mlx-examples directory, sync fork:
# https://github.com/LLMAppArchitect/mlx-lm/tree/main

git pull 
cd llms

# use conda env:
conda activate m3mlx

# install deps:
pip install -e .  


```

Install log:

``` 
Looking in indexes: https://bytedpypi.byted.org/simple/
Obtaining file:///Users/bytedance/ai/mlx-lm/llms
  Preparing metadata (setup.py) ... done
Requirement already satisfied: mlx>=0.17.0 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from mlx-lm==0.19.1) (0.18.0)
Requirement already satisfied: numpy in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from mlx-lm==0.19.1) (1.26.4)
Requirement already satisfied: transformers>=4.39.3 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (4.41.1)
Requirement already satisfied: protobuf in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from mlx-lm==0.19.1) (5.27.0)
Requirement already satisfied: pyyaml in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from mlx-lm==0.19.1) (6.0.1)
Requirement already satisfied: jinja2 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from mlx-lm==0.19.1) (3.1.4)
Requirement already satisfied: filelock in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (3.14.0)
Requirement already satisfied: huggingface-hub<1.0,>=0.23.0 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (0.23.4)
Requirement already satisfied: packaging>=20.0 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (24.0)
Requirement already satisfied: regex!=2019.12.17 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (2024.5.15)
Requirement already satisfied: requests in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (2.32.2)
Requirement already satisfied: tokenizers<0.20,>=0.19 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (0.19.1)
Requirement already satisfied: safetensors>=0.4.1 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (0.4.3)
Requirement already satisfied: tqdm>=4.27 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (4.66.4)
Requirement already satisfied: sentencepiece!=0.1.92,>=0.1.91 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (0.2.0)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from jinja2->mlx-lm==0.19.1) (2.1.5)
Requirement already satisfied: fsspec>=2023.5.0 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.23.0->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (2024.5.0)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.23.0->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (4.12.0)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (2.2.1)
Requirement already satisfied: certifi>=2017.4.17 in /opt/miniconda3/envs/m3mlx/lib/python3.11/site-packages (from requests->transformers>=4.39.3->transformers[sentencepiece]>=4.39.3->mlx-lm==0.19.1) (2024.2.2)

Installing collected packages: mlx-lm
  Attempting uninstall: mlx-lm
    Found existing installation: mlx-lm 0.18.2
    Uninstalling mlx-lm-0.18.2:
      Successfully uninstalled mlx-lm-0.18.2
  Running setup.py develop for mlx-lm
Successfully installed mlx-lm-0.19.1

```


## Run MXL LLM Server

```shell
cd llms/mlx_lm
```

Start the server with:

> see: [SERVER.md](llms%2Fmlx_lm%2FSERVER.md)

```shell
mlx_lm.server --model <path_to_model_or_hf_repo>
```

For example:

```shell
# https://huggingface.co/mlx-community/Ministral-8B-Instruct-2410-8bit
mlx_lm.server --model mlx-community/Ministral-8B-Instruct-2410-8bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/Qwen2.5-Coder-14B-Instruct-8bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-8bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/Mistral-Nemo-Instruct-2407-8bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit --trust-remote-code --port 8722
mlx_lm.server --model mlx-community/internlm2_5-7b-chat-8bit --trust-remote-code --port 8722

# for run
run:
	mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-8bit --trust-remote-code --port 8722
	#mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit --trust-remote-code --port 8722
	#mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-8bit --trust-remote-code --port 8722
	#mlx_lm.server --model mlx-community/Qwen2.5-14B-Instruct-4bit --trust-remote-code --port 8722
	#mlx_lm.server --model mlx-community/Ministral-8B-Instruct-2410-4bit --trust-remote-code --port 8722
	#mlx_lm.server --model mlx-community/Ministral-8B-Instruct-2410-8bit --trust-remote-code --port 8722
```

This will start a text generation server on port `8080` of the `localhost`
using Mistral 7B instruct. The model will be downloaded from the provided
Hugging Face repo if it is not already in the local cache.

To see a full list of options run:

```shell
mlx_lm.server --help
```

You can make a request to the model by running:

```shell
curl localhost:8722/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7,
     "max_tokens": 100,
   }'
```

output:

```json 
{
  "id": "chatcmpl-74e66064-8727-411a-ada3-d5287b2c83a2",
  "system_fingerprint": "fp_73a731bd-bd00-4dcd-8fac-8f3f452210a2",
  "object": "chat.completions",
  "model": "default_model",
  "created": 1721634359,
  "choices": [
    {
      "index": 0,
      "logprobs": {
        "token_logprobs": [
          -2.4453125,
          -1.28125,
          -1.421875,
          -0.25,
          -7.53125,
          -1.15625,
          -4.09375,
          -0.390625,
          -3.0625,
          -0.84375,
          -2.53125,
          -0.125,
          -0.40625,
          -0.015625,
          -0.15625,
          -0.265625,
          -1.015625,
          -1.6484375,
          -1.0625,
          -0.40625,
          -4.390625,
          -0.296875,
          -1.078125,
          -3.0625,
          -0.328125,
          -0.21875,
          -0.390625,
          -2.015625,
          -3.46875,
          0.0,
          -0.765625,
          -2.609375,
          -1.921875,
          -1.078125,
          -1.859375,
          -1.625,
          -0.09375,
          -0.015625,
          -1.5625,
          -2.1015625,
          -1.65625,
          -0.21875,
          0.0,
          0.0,
          -1.640625,
          -0.0625,
          0.0,
          -1.234375,
          -0.6875,
          -0.53125,
          -0.078125,
          -0.03125,
          -1.015625,
          -0.109375,
          -3.4765625,
          -0.015625,
          -2.140625,
          -1.34375,
          -1.0625,
          -2.21875,
          -1.046875,
          -0.046875,
          -0.375,
          -1.0,
          -1.0625,
          -3.21875,
          -0.5,
          -0.234375,
          -0.15625,
          -2.015625,
          -1.265625,
          -0.390625,
          -2.265625,
          -0.0625,
          -1.59375,
          -3.5625,
          -0.59375,
          -0.46875,
          -1.0,
          -1.3515625,
          -0.296875,
          -1.4375,
          0.0,
          -1.1875,
          -0.46875,
          -0.15625,
          -0.375,
          -0.0625,
          -0.0625,
          -3.90625,
          -0.9375,
          -0.5625,
          -0.25,
          -2.53125,
          -0.28125,
          -2.640625,
          -0.59375,
          -0.75,
          -0.53125,
          -0.71875
        ],
        "top_logprobs": [],
        "tokens": [
          39584,
          346,
          5846,
          725,
          3716,
          489,
          4330,
          25341,
          16375,
          3103,
          1226,
          725,
          395,
          3556,
          1593,
          43916,
          465,
          2423,
          57436,
          334,
          19109,
          446,
          395,
          16375,
          22006,
          55098,
          465,
          53057,
          51040,
          334,
          465,
          848,
          285,
          3235,
          53057,
          4144,
          334,
          465,
          461,
          2423,
          57436,
          830,
          285,
          3235,
          5168,
          334,
          465,
          461,
          2136,
          505,
          395,
          1420,
          17338,
          465,
          312,
          281,
          5128,
          285,
          2423,
          5128,
          1883,
          938,
          334,
          55098,
          10363,
          6069,
          410,
          1420,
          328,
          410,
          2863,
          46301,
          2119,
          517,
          2014,
          334,
          4872,
          285,
          3235,
          2423,
          11740,
          334,
          465,
          29581,
          560,
          410,
          1420,
          4736,
          505,
          6662,
          12590,
          281,
          1239,
          1377,
          3089,
          22865,
          560,
          810,
          6025,
          3328
        ]
      },
      "finish_reason": "length",
      "message": {
        "role": "assistant",
        "content": "Sure! Here's What I'd Say Given That It's a Test:\n\n---\n\n**Test Scenario: Validation of a Given Statement**\n\n**Scenario Outline:** \n- **Scenario Name:** \"Test Scenario\"\n- **Description:** \"This is a test!\"\n\n**1. Pre-Test Preparations:**\n\nBefore starting the test, the following preparations must be made: \n\n- **Test Environment:** Ensure that the test environment is setup correctly. This may include ensuring that all necessary software"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 100,
    "total_tokens": 116
  }
}

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

- `logit_bias`: (Optional) A dictionary mapping token IDs to their bias
  values. Defaults to `None`.

- `logprobs`: (Optional) An integer specifying the number of top tokens and
  corresponding log probabilities to return for each output in the generated
  sequence. If set, this can be any value between 1 and 10, inclusive.



### Text Models 

- [MLX LM](llms/README.md) a package for LLM text generation, fine-tuning, and more.
- [Transformer language model](transformer_lm) training.
- Minimal examples of large scale text generation with [LLaMA](llms/llama),
  [Mistral](llms/mistral), and more in the [LLMs](llms) directory.
- A mixture-of-experts (MoE) language model with [Mixtral 8x7B](llms/mixtral).
- Parameter efficient fine-tuning with [LoRA or QLoRA](lora).
- Text-to-text multi-task Transformers with [T5](t5).
- Bidirectional language understanding with [BERT](bert).

### Image Models 

- Image classification using [ResNets on CIFAR-10](cifar).
- Generating images with [Stable Diffusion or SDXL](stable_diffusion).
- Convolutional variational autoencoder [(CVAE) on MNIST](cvae).

### Audio Models

- Speech recognition with [OpenAI's Whisper](whisper).

### Multimodal models

- Joint text and image embeddings with [CLIP](clip).
- Text generation from image and text inputs with [LLaVA](llava).

### Other Models 

- Semi-supervised learning on graph-structured data with [GCN](gcn).
- Real NVP [normalizing flow](normalizing_flow) for density estimation and
  sampling.

### Hugging Face

Note: You can now directly download a few converted checkpoints from the [MLX
Community](https://huggingface.co/mlx-community) organization on Hugging Face.
We encourage you to join the community and [contribute new
models](https://github.com/ml-explore/mlx-examples/issues/155).

## Contributing 

We are grateful for all of [our
contributors](ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX Examples and wish to be acknowledged, please add your name to the list in your
pull request.

## Citing MLX Examples

The MLX software suite was initially developed with equal contribution by Awni
Hannun, Jagrit Digani, Angelos Katharopoulos, and Ronan Collobert. If you find
MLX Examples useful in your research and wish to cite it, please use the following
BibTex entry:

```
@software{mlx2023,
  author = {Awni Hannun and Jagrit Digani and Angelos Katharopoulos and Ronan Collobert},
  title = {{MLX}: Efficient and flexible machine learning on Apple silicon},
  url = {https://github.com/ml-explore},
  version = {0.0},
  year = {2023},
}
```
