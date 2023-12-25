import argparse

import numpy
from transformers import BertModel


def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key


def convert(bert_model: str, mlx_model: str) -> None:
    model = BertModel.from_pretrained(bert_model)
    # save the tensors
    tensors = {
        replace_key(key): tensor.numpy() for key, tensor in model.state_dict().items()
    }
    numpy.savez(mlx_model, **tensors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BERT weights to MLX.")
    parser.add_argument(
        "--bert-model",
        choices=[
            "bert-base-uncased",
            "bert-base-cased",
            "bert-large-uncased",
            "bert-large-cased",
        ],
        default="bert-base-uncased",
        help="The huggingface name of the BERT model to save.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/bert-base-uncased.npz",
        help="The output path for the MLX BERT weights.",
    )
    args = parser.parse_args()

    convert(args.bert_model, args.mlx_model)
