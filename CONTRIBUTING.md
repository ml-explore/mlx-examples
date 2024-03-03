# Contributing to mlx-examples

We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests

1. Fork and submit pull requests to the repo. 
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests and at least one review. 
4. For code formatting install `pre-commit` using something like `pip install pre-commit` and run `pre-commit install`.
   This should install hooks for running `black` and `clang-format` to ensure
   consistent style for C++ and python code.
 
   You can also run the formatters manually as follows:
 
     ```bash
     clang-format -i file.cpp
     ```
 
     ```bash
     black file.py
     ```
 
   or run `pre-commit run --all-files` to check all files in the repo.

## Tips on porting LLMs from HuggingFace

First make sure model you want to port has weights in [safetensor](https://huggingface.co/docs/safetensors/index) format, if not you can use convert using [this](https://huggingface.co/spaces/safetensors/convert) tool.

After that it's relatively simple, you need to add the model file to `llms/mlx_lm/models` directory 

You can see other examples in there, we recommend starting from one of the models that is pretty similar to model you want to add, as most models are based on LLAMA/Mistral architechture.

Make sure the name of the model file is the same as the `model_type` in the `config` for example [starcoder2](https://huggingface.co/bigcode/starcoder2-7b/blob/main/config.json#L17) 

For model layer names we suggest either referring to the transformers implementation if you are familiar with the codebase, or loading the model weights and checking the weight names which will give you a hint of the model structure, you can take a look at the names of the weights by inspecting `model.safetensors.index.json`

If you want to add LoRA support you can edit [`mlx_lm/tuner/utils.py`](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/utils.py#L27-L60)

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to mlx-examples, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
