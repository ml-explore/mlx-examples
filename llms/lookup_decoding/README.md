# Prompt Lookup Decoding
This example implements [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) for LLM text generation. This particular example uses [Mistral](/llms/mistral/) as the model but that can be changed with minimal adjustments to `decoder.py`. Prompt lookup decoding is a variation of [speculative decoding](https://arxiv.org/abs/2211.17192) where the draft model has been replaced with a simple prompt lookup function to generate the draft. For *input-grounded* tasks such as summarization, document QA or code editing, this technique can provide significant speedups with no effect on output quality.

### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download a **Mistral** model and tokenizer:

```
curl -O https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

Then, convert the weights with:

```
python convert.py --torch-path <path_to_torch>
```

To generate a 4-bit quantized model, use ``-q``. For a full list of options:

```
python convert.py --help
```

By default, the conversion script will make the directory `mlx_model` and save
the converted `weights.npz`, `tokenizer.model`, and `config.json` there.

> [!TIP]
> Alternatively, you can also download a few converted checkpoints from the
> [MLX Community](https://huggingface.co/mlx-community) organization on Hugging
> Face and skip the conversion step.

### Run
```
python decoder.py --prompt "[INST] Repeat the following phrase 10 times: 'The quick brown fox jumps over the lazy dog.'. Don't say antyhing else. [/INST]" --max-tokens 250
```

Alternatively you can call the API

```python
from decoder import PromptLookupDecoder

prompt = "[INST] Repeat the following phrase 10 times: 'The quick brown fox jumps over the lazy dog.'. Don't say antyhing else. [/INST] "

engine = PromptLookupDecoder("mlx_model")

engine.prompt_lookup(prompt=prompt, max_tokens=250, n_draft=10, 
    ngram_max=3, ngram_min=1, temp=0.0, seed=0, color=True)
```
