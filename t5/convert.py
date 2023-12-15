from transformers import T5ForConditionalGeneration
import numpy as np


SHARED_REPLACEMENT_PATTERNS = [
    (".block.", ".layers."),
    (".k.", ".key_proj."),
    (".o.", ".out_proj."),
    (".q.", ".query_proj."),
    (".v.", ".value_proj."),
    ("shared.", "wte."),
    (".layer.0.layer_norm.", ".ln1."),
    (".layer.1.layer_norm.", ".ln2."),
    (".layer.2.layer_norm.", ".ln3."),
    (".final_layer_norm.", ".ln."),
    (
        ".relative_attention_bias.",
        ".relative_attention_bias.embeddings."
    ),
]

ENCODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".attention."),
    (".layer.1.DenseReluDense.wi.", ".linear1."),
    (".layer.1.DenseReluDense.wo.", ".linear2."),
]

DECODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".self_attention."),
    (".layer.1.EncDecAttention.", ".cross_attention."),
    (".layer.2.DenseReluDense.wi.", ".linear1."),
    (".layer.2.DenseReluDense.wo.", ".linear2."),
]

def replace_key(key: str) -> str:
    for old, new in SHARED_REPLACEMENT_PATTERNS:
        key = key.replace(old, new)
    if key.startswith("encoder."):
        for old, new in ENCODER_REPLACEMENT_PATTERNS:
            key = key.replace(old, new)
    elif key.startswith("decoder."):
        for old, new in DECODER_REPLACEMENT_PATTERNS:
            key = key.replace(old, new)
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
