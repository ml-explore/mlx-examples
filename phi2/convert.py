from transformers import AutoModelForCausalLM

import numpy


def split_attention_matrix(state_dict, key) -> dict:
    # "transformer.h.0.mixer"
    _, model_dim = state_dict[key + ".weight"].shape
    # (3 * model_dim, model_dim)
    Wqkv_weight_key = key + ".weight"
    Wq_weight = state_dict[Wqkv_weight_key][:model_dim, :]
    Wk_weight = state_dict[Wqkv_weight_key][model_dim : 2 * model_dim, :]
    Wv_weight = state_dict[Wqkv_weight_key][2 * model_dim :, :]

    # (3 * model_dim)
    Wqkv_bias_key = key + ".bias"
    Wq_bias = state_dict[Wqkv_bias_key][:model_dim]
    Wk_bias = state_dict[Wqkv_bias_key][model_dim : 2 * model_dim]
    Wv_bias = state_dict[Wqkv_bias_key][2 * model_dim :]

    out_key = key.replace("mixer.Wqkv", "self_attention")

    return {
        out_key + ".query_proj.weight": Wq_weight,
        out_key + ".query_proj.bias": Wq_bias,
        out_key + ".key_proj.weight": Wk_weight,
        out_key + ".key_proj.bias": Wk_bias,
        out_key + ".value_proj.weight": Wv_weight,
        out_key + ".value_proj.bias": Wv_bias,
    }


def replace_key(key: str) -> str:
    if "wte.weight" in key:
        key = "wte.weight"

    if ".mlp" in key:
        key = key.replace(".mlp", "")

    if ".mixer.out_proj" in key:
        key = key.replace(".mixer", ".self_attention")

    return key


def convert():
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    state_dict = model.state_dict()
    keys = list(state_dict.keys())

    for key in keys:
        if ".mixer.Wqkv.weight" not in key:
            continue
        key_stub = key.rstrip(".weight")
        state_dict.update(split_attention_matrix(state_dict, key_stub))

        del state_dict[key_stub + ".weight"]
        del state_dict[key_stub + ".bias"]

    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    numpy.savez("weights/phi-2.npz", **weights)


if __name__ == "__main__":
    convert()
