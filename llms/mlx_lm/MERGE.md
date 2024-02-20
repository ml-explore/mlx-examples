# Model Merging

You can use `mlx-lm` to merge models and upload them to the Hugging
Face hub or save them locally for LoRA fine tuning.

The main command is `mlx_lm.merge`:

```shell
python -m mlx_lm.merge --config config.yaml 
```

The merged model will be saved by default in `mlx_merged_model`. To see a
full list of options run:

```shell
python -m mlx_lm.merge --help
```

Here is an example `config.yaml`:

```yaml
models:
  - OpenPipe/mistral-ft-optimized-1218
  - mlabonne/NeuralHermes-2.5-Mistral-7B
method: slerp
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
```

The `models` field is a list of Hugging Face repo ids. The first model in the
list is treated as the base model into which the remaining models are merged.

The `method` field is the merging method. Right now `slerp` is the only
supported method.

The `parameters` are the corresponding parameters for the given `method`.
Each parameter is a list with `filter` determining which layer the parameter
applies to and `value` determining the actual value used. The last item in
the list without a `filter` field is the default.

If `value` is a list, it specifies the start and end values for the
corresponding segment of blocks. In the example above, the models have 32
blocks. For blocks 1-8, the layers with `self_attn` in the name will use the
values `np.linspace(0, 0.5, 8)`, the same layers in the next 8 blocks (9-16)
will use `np.linspace(0.5, 0.3, 8)`, and so on.
