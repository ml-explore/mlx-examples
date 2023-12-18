from transformers import T5ForConditionalGeneration
import numpy as np


SHARED_REPLACEMENT_PATTERNS = [
    (".block.", ".layers."),
    (".k.", ".key_proj."),
    (".o.", ".out_proj."),
    (".q.", ".query_proj."),
    (".v.", ".value_proj."),
    ("shared.", "wte."),
    ("lm_head.", "lm_head.linear."),
    (".layer.0.layer_norm.", ".ln1."),
    (".layer.1.layer_norm.", ".ln2."),
    (".layer.2.layer_norm.", ".ln3."),
    (".final_layer_norm.", ".ln."),
    (
        "layers.0.layer.0.SelfAttention.relative_attention_bias.",
        "relative_attention_bias.embeddings.",
    ),
]

ENCODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".attention."),
    (".layer.1.DenseReluDense.", ".dense."),
]

DECODER_REPLACEMENT_PATTERNS = [
    (".layer.0.SelfAttention.", ".self_attention."),
    (".layer.1.EncDecAttention.", ".cross_attention."),
    (".layer.2.DenseReluDense.", ".dense."),
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


def convert(model_name):
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto")
    weights = {
        replace_key(k): v.numpy().astype(np.float16)
        for k, v in model.state_dict().items()
    }
    file_name = model_name.replace("/", "-")
    np.savez(f"{file_name}.npz", **weights)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert T5 weights to MLX")
    parser.add_argument(
        "--model",
        type=str,
        help="Name of the T5 model.",
        choices=[
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-3b",
            "t5-11b",
            "google/flan-t5-small",
            "google/flan-t5-base",
            "google/flan-t5-large",
            "google/flan-t5-xl",
            "google/flan-t5-xxl",
            "google/flan-t5-ul2",
        ],
        default="t5-small",
    )
    args = parser.parse_args()
    convert(args.model)
