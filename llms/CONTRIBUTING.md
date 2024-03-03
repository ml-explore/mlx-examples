# Contributing to mlx-examples-llms

We want to make contributing to this project as easy and transparent as
possible. For general contributing guide please see this [guide](https://github.com/ml-explore/mlx-examples/blob/main/CONTRIBUTING.md). Below are some tips on porting LLMs from HuggingFace.

Before you start make sure you have an editable install `pip install -e .`

Then check if the model you want to port has weights in [safetensor](https://huggingface.co/docs/safetensors/index) format, if not you can use convert using [this](https://huggingface.co/spaces/safetensors/convert) tool.

After that it's relatively simple, you need to add the model file to [`mlx_lm/models`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm/models) directory. 

You can see other examples in there, we recommend starting from one of the models that is pretty similar to model you want to add, as most models are based on LLAMA/Mistral architechture.

Make sure the name of the model file is the same as the `model_type` in the `config` for example [starcoder2](https://huggingface.co/bigcode/starcoder2-7b/blob/main/config.json#L17) 

For model layer names we suggest either referring to the transformers implementation if you are familiar with the codebase, or loading the model weights and checking the weight names which will give you a hint of the model structure, you can take a look at the names of the weights by inspecting `model.safetensors.index.json`

If you want to add LoRA support you can edit [`mlx_lm/tuner/utils.py`](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/utils.py#L27-L60)