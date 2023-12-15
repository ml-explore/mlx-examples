from transformers import T5ForConditionalGeneration
import numpy as np

def replace_key(key: str) -> str:
    key = key.replace(".block.", ".layers.")
    key = key.replace(".layer.0.SelfAttention.", ".attention.")
    key = key.replace(".k.", ".key_proj.")
    key = key.replace(".o.", ".out_proj.")
    key = key.replace(".q.", ".query_proj.")
    key = key.replace(".v.", ".value_proj.")
    key = key.replace(".layer.0.layer_norm.", ".ln1.")
    key = key.replace(".layer.1.layer_norm.", ".ln2.")
    key = key.replace(".layer.1.DenseReluDense.wi.", ".linear1.")
    key = key.replace(".layer.1.DenseReluDense.wo.", ".linear2.")
    key = key.replace(".final_layer_norm.", ".ln.")
    key = key.replace("shared.", "wte.")
    key = key.replace("encoder.layers.0.attention.relative_attention_bias.",
                      "position_bias.relative_attention_bias.")
    return key

def convert():
    model = T5ForConditionalGeneration.from_pretrained(
        "t5-small", torch_dtype="auto"
    )
    state_dict = model.state_dict()
    weights = {replace_key(k): v.numpy() for k, v in state_dict.items()}
    np.savez("weights.npz", **weights)


if __name__ == "__main__":
    convert()
