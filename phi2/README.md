# Phi-2

Phi-2 is a 2.7B parameter model released by Microsoft and trained on a mixture of GPT-4 outputs and clean web-text.
Its performance theoretically rivals much, much stronger models.

## Downloading and Converting Weights

To download and convert the model:

```sh 
python phi2/convert.py
```

That will fill in `weights/phi-2.npz`.

## Running the Model

🚧 (Not yet done) To run the model:

```sh
python phi2/generate.py
```

Layer-by-layer forward pass outputs are currently shown in the outputs.txt files.
