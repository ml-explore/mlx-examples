import numpy as np
from transformers import AutoModelForCausalLM


def replace_key(key: str) -> str:
    if "wte.weight" in key:
        key = "wte.weight"

    if ".mlp" in key:
        key = key.replace(".mlp", "")
    return key


def convert():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    np.savez("weights.npz", **weights)


if __name__ == "__main__":
    convert()
