import argparse
from typing import List

import model
import numpy as np
from transformers import AutoModel, AutoTokenizer


def run_torch(bert_model: str, batch: List[str]):
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    torch_model = AutoModel.from_pretrained(bert_model)
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    torch_forward = torch_model(**torch_tokens)
    torch_output = torch_forward.last_hidden_state.detach().numpy()
    torch_pooled = torch_forward.pooler_output.detach().numpy()
    return torch_output, torch_pooled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a BERT-like model for a batch of text."
    )
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="The model identifier for a BERT-like model from Hugging Face Transformers.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default="weights/bert-base-uncased.npz",
        help="The path of the stored MLX BERT weights (npz file).",
    )
    parser.add_argument(
        "--text",
        nargs="+",
        default=["This is an example of BERT working in MLX."],
        help="A batch of texts to process. Multiple texts should be separated by spaces.",
    )

    args = parser.parse_args()

    torch_output, torch_pooled = run_torch(args.bert_model, args.text)

    mlx_output, mlx_pooled = model.run(args.bert_model, args.mlx_model, args.text)

    if torch_pooled is not None and mlx_pooled is not None:
        assert np.allclose(
            torch_output, mlx_output, rtol=1e-4, atol=1e-5
        ), "Model output is different"
        assert np.allclose(
            torch_pooled, mlx_pooled, rtol=1e-4, atol=1e-5
        ), "Model pooled output is different"
        print("Tests pass :)")
    else:
        print("Pooled outputs were not compared due to one or both being None.")
