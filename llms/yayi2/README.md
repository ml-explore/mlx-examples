# YAYI2

YAYI 2 is a collection of open-source large language models launched by Wenge Technology. YAYI2-30B is a Transformer-based large language model, and has been pretrained for 2.65 trillion tokens of multilingual data with high quality. The base model is aligned with human values through supervised fine-tuning with millions of instructions and reinforcement learning from human feedback (RLHF).


### Setup

Install the dependencies:

```
pip install -r requirements.txt
```

Next, download and convert the model. 

```sh
python convert.py --hf-path <path_to_huggingface_model>
```

To generate a 4-bit quantized model, use `-q`. For a full list of options run:

```
python convert.py --help
```

The converter downloads the model from Hugging Face. The default model is
`wenge-research/yayi2-30b`. Check out the [Hugging Face
page](https://huggingface.co/wenge-research) to see a list of available models.

By default, the conversion script will save the converted `weights.npz`,
tokenizer, and `config.json` in the `mlx_model` directory.

### Run

Once you've converted the weights, you can interact with the Yayi2
model:

```
python yayi.py --prompt "The winter in Beijing is"
```


