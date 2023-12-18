from transformers import AutoModelForCausalLM
import numpy as np


def replace_key(key: str) -> str:
    if key.startswith("transformer."):
        # remove transformer prefix
        key = key.replace("transformer.", "")

    return key


def convert():
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-1_8B", trust_remote_code=True
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    np.savez("weights.npz", **weights)


if __name__ == "__main__":
    convert()
