# Managing Models

You use `mlx-lm` to manage models downloaded locally in your machine. They are stored in 
Hugging Face cache.

Scan Models: 

```shell
mlx_lm.model --scan-models 
```

Get details on a specific model:
```shell
mlx_lm.model --model mlx-community/Mistral-7B-Instruct-v0.2-4bit
```

Delete a model by name:
```shell
mlx_lm.model --delete-model mlx-community/Mistral-7B-Instruct-v0.2-4bit
```