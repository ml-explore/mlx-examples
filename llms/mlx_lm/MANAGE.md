# Managing Models

You can use `mlx-lm` to manage models downloaded locally in your machine. They
are stored in the Hugging Face cache.

Scan models: 

```shell
mlx_lm.manage --scan
```

Specify a `--pattern` to get info on a single or specific set of models:

```shell
mlx_lm.manage --scan --pattern mlx-community/Mistral-7B-Instruct-v0.2-4bit
```

To delete a model (or multiple models):

```shell
mlx_lm.manage --delete --pattern mlx-community/Mistral-7B-Instruct-v0.2-4bit
```
