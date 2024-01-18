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
    bert_model = "bert-base-uncased"
    mlx_model = "weights/bert-base-uncased.npz"
    batch = [
        "This is an example of BERT working in MLX.",
        "A second string",
        "This is another string.",
    ]
    torch_output, torch_pooled = run_torch(bert_model, batch)
    mlx_output, mlx_pooled = model.run(bert_model, mlx_model, batch)
    assert np.allclose(
        torch_output, mlx_output, rtol=1e-4, atol=1e-5
    ), "Model output is different"
    assert np.allclose(
        torch_pooled, mlx_pooled, rtol=1e-4, atol=1e-5
    ), "Model pooled output is different"
    print("Tests pass :)")
