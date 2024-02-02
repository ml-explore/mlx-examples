# Transformer LM 

This is an example of a decoder-only Transformer LM. The only dependency is
MLX. 

Run the example on the GPU with:

```
python main.py --gpu
```

By default the dataset is the [PTB corpus](https://paperswithcode.com/dataset/penn-treebank). Choose a different dataset with the `--dataset` option.
