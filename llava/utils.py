import mlx.core as mx
import torch
from typing import Tuple


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(
        getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))


def should_keep_weight(key: str):
    return not ("position_ids" in key)


def map_vision_tower_weights(key: str, value: torch.Tensor) -> Tuple[str, torch.Tensor]:
    key = key.replace("embeddings.", "")
    key = key.replace("encoder.", "")
    key = key.replace("position_embedding.weight", "position_embedding")

    key = key.replace('vision_model.', '')

    # Map attention layers
    if "self_attn." in key:
        key = key.replace("self_attn.", "attention.")
    if "q_proj." in key:
        key = key.replace("q_proj.", "query_proj.")
    if "k_proj." in key:
        key = key.replace("k_proj.", "key_proj.")
    if "v_proj." in key:
        key = key.replace("v_proj.", "value_proj.")
    if "layer_norm1." in key:
        key = key.replace("layer_norm1.", "ln1.")
    if "layer_norm2." in key:
        key = key.replace("layer_norm2.", "ln2.")
    # Map ffn layers
    if "mlp.fc1" in key:
        key = key.replace("mlp.fc1", "linear1")
    if "mlp.fc2" in key:
        key = key.replace("mlp.fc2", "linear2")
    # Fix layernorm typo
    if "pre_layrnorm" in key:
        # Fix typo in weights :)
        key = key.replace("pre_layrnorm", "pre_layernorm")
    if "patch_embedding.weight" in key:
        # Initially, value: [out_channels, in_channels, kH, KW].
        # We want [out_channels, kH, KW, in_channels]
        value = value.permute(0, 2, 3, 1)
    return (key, value)


def map_language_model_weights(key: str, value: torch.Tensor) -> Tuple[str, torch.Tensor]:
    key = key.replace('language_model.model.', 'language_model.')
    key = key.replace('mlp.', 'feed_forward.')
    key = key.replace("down_proj", "w2")
    key = key.replace("up_proj", "w3")
    key = key.replace("gate_proj", "w1")
    key = key.replace("input_layernorm", "attention_norm")
    key = key.replace("post_attention_layernorm", "ffn_norm")
    key = key.replace("lm_head", "output")

    key = key.replace("embed_tokens", "tok_embeddings")
    key = key.replace("self_attn", "attention")

    key = key.replace("q_proj", "wq")
    key = key.replace("k_proj", "wk")
    key = key.replace("v_proj", "wv")
    key = key.replace("o_proj", "wo")

    return (key, value)


def map_multi_modal_projector_weights(key: str, value: torch.Tensor) -> Tuple[str, torch.Tensor]:
    return (key, value)


def map_weights(key: str, value: torch.Tensor) -> Tuple[str, mx.array]:

    if 'vision_tower' in key:
        key, value = map_vision_tower_weights(key, value)
    elif 'language_model' in key:
        key, value = map_language_model_weights(key, value)
    elif 'multi_modal_projector' in key:
        key, value = map_multi_modal_projector_weights(key, value)

    return (key, torch_to_mx(value, dtype=str(value.dtype).replace("torch.", "")))
