# Contributing to MLX LM 

Below are some tips to port LLMs available on Hugging Face to MLX.

Before starting checkout the [general contribution
guidelines](https://github.com/ml-explore/mlx-examples/blob/main/CONTRIBUTING.md).

Next, from this directory, do an editable install:

```shell
pip install -e .
```

Then check if the model has weights in the
[safetensors](https://huggingface.co/docs/safetensors/index) format. If not
[follow instructions](https://huggingface.co/spaces/safetensors/convert) to
convert it.

After that, add the model file to the
[`mlx_lm/models`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm/models)
directory. You can see other examples there. We recommend starting from a model
that is similar to the model you are porting.

Make sure the name of the new model file is the same as the `model_type` in the
`config.json`, for example
[starcoder2](https://huggingface.co/bigcode/starcoder2-7b/blob/main/config.json#L17).

To determine the model layer names, we suggest either:

- Refer to the Transformers implementation if you are familiar with the
  codebase.
- Load the model weights and check the weight names which will tell you about
  the model structure.
- Look at the names of the weights by inspecting `model.safetensors.index.json`
  in the Hugging Face repo.

To add LoRA support edit
[`mlx_lm/tuner/utils.py`](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/tuner/utils.py#L27-L60)

Finally, add a test for the new modle type to the [model
tests](https://github.com/ml-explore/mlx-examples/blob/main/llms/tests/test_models.py).

From the `llms/` directory, you can run the tests with:

```shell
python -m unittest discover tests/
```
