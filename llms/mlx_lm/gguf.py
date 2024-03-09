import glob
import json
from enum import IntEnum
from pathlib import Path
from typing import Iterable

import mlx.core as mx
from mlx.utils import tree_flatten
from sentencepiece import SentencePieceProcessor

from .utils import get_model_path, load_model


class TokenType(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class GGMLFileType(IntEnum):
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2


class SentencePieceVocab:
    def __init__(self, fname_tokenizer: Path, fname_added_tokens: Path | None) -> None:
        self.sentencepiece_tokenizer = SentencePieceProcessor(str(fname_tokenizer))
        added_tokens: dict[str, int]
        if fname_added_tokens is not None:
            added_tokens = json.load(open(fname_added_tokens, encoding="utf-8"))
        else:
            added_tokens = {}

        vocab_size: int = self.sentencepiece_tokenizer.vocab_size()

        new_tokens = {
            id: piece for piece, id in added_tokens.items() if id >= vocab_size
        }
        expected_new_ids = list(range(vocab_size, vocab_size + len(new_tokens)))
        actual_new_ids = sorted(new_tokens.keys())

        if expected_new_ids != actual_new_ids:
            raise ValueError(
                f"Expected new token IDs {expected_new_ids} to be sequential; got {actual_new_ids}"
            )

        # Token pieces that were added to the base vocabulary.
        self.added_tokens_dict = added_tokens
        self.added_tokens_list = [new_tokens[id] for id in actual_new_ids]
        self.vocab_size_base = vocab_size
        self.vocab_size = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def sentencepiece_tokens(self) -> Iterable[tuple[bytes, float, TokenType]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            piece = tokenizer.id_to_piece(i)
            text: bytes = piece.encode("utf-8")
            score: float = tokenizer.get_score(i)

            toktype = TokenType.NORMAL
            if tokenizer.is_unknown(i):
                toktype = TokenType.UNKNOWN
            if tokenizer.is_control(i):
                toktype = TokenType.CONTROL

            # NOTE: I think added_tokens are user defined.
            # ref: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
            # if tokenizer.is_user_defined(i): toktype = TokenType.USER_DEFINED

            if tokenizer.is_unused(i):
                toktype = TokenType.UNUSED
            if tokenizer.is_byte(i):
                toktype = TokenType.BYTE

            yield text, score, toktype

    def added_tokens(self) -> Iterable[tuple[bytes, float, TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, TokenType.USER_DEFINED

    def all_tokens(self) -> Iterable[tuple[bytes, float, TokenType]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

    @staticmethod
    def load(path: Path) -> "SentencePieceVocab":
        added_tokens_path = path.parent / "added_tokens.json"
        return SentencePieceVocab(
            path, added_tokens_path if added_tokens_path.exists() else None
        )


def translate_weight_names(name):
    name = name.replace("model.layers.", "blk.")
    name = name.replace("mlp.gate_proj", "ffn_gate")
    name = name.replace("mlp.down_proj", "ffn_down")
    name = name.replace("mlp.up_proj", "ffn_up")
    name = name.replace("self_attn.q_proj", "attn_q")
    name = name.replace("self_attn.k_proj", "attn_k")
    name = name.replace("self_attn.v_proj", "attn_v")
    name = name.replace("self_attn.o_proj", "attn_output")
    name = name.replace("input_layernorm", "attn_norm")
    name = name.replace("post_attention_layernorm", "ffn_norm")
    name = name.replace("model.embed_tokens", "token_embd")
    name = name.replace("model.norm", "output_norm")
    name = name.replace("lm_head", "output")
    return name


def load_weights(model_path: str):
    model_path = get_model_path(model_path)
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


model_path = get_model_path("mistralai/Mistral-7B-v0.1")

# model = load_model(model_path)
weights = load_weights(model_path)

with open(model_path / "config.json", "r") as f:
    config = json.load(f)

# weights = dict(tree_flatten(model.parameters()))

# rename weights for gguf format
weights = {translate_weight_names(k): v for k, v in weights.items()}


if not (model_path / "tokenizer.model").exists():
    raise ValueError("Tokenizer model not found")

vocab = SentencePieceVocab.load(model_path / "tokenizer.model")

metadata = {
    "general.name": "llama",
    "llama.context_length": (
        mx.array(config["max_position_embeddings"], dtype=mx.uint32)
        if config.get("max_position_embeddings") is not None
        else None
    ),
    "llama.embedding_length": (
        mx.array(config["hidden_size"], dtype=mx.uint32)
        if config.get("hidden_size") is not None
        else None
    ),
    "llama.block_count": (
        mx.array(config["num_hidden_layers"], dtype=mx.uint32)
        if config.get("num_hidden_layers") is not None
        else None
    ),
    "llama.feed_forward_length": (
        mx.array(config["intermediate_size"], dtype=mx.uint32)
        if config.get("intermediate_size") is not None
        else None
    ),
    "llama.rope.dimension_count": (
        mx.array(
            config["hidden_size"] // config["num_attention_heads"], dtype=mx.uint32
        )
        if config.get("hidden_size") is not None
        and config.get("num_attention_heads") is not None
        else None
    ),
    "llama.attention.head_count": (
        mx.array(config["num_attention_heads"], dtype=mx.uint32)
        if config.get("num_attention_heads") is not None
        else None
    ),
    "llama.attention.head_count_kv": (
        mx.array(
            config.get("num_key_value_heads", config["num_attention_heads"]),
            dtype=mx.uint32,
        )
        if config.get("num_attention_heads") is not None
        else None
    ),
    "llama.expert_count": (
        mx.array(config.get("num_local_experts", None), dtype=mx.uint32)
        if config.get("num_local_experts") is not None
        else None
    ),
    "llama.expert_used_count": (
        mx.array(config.get("num_experts_per_tok", None), dtype=mx.uint32)
        if config.get("num_experts_per_tok") is not None
        else None
    ),
    "llama.attention.layer_norm_rms_epsilon": (
        mx.array(config.get("rms_norm_eps", 1e-05))
    ),
    "llama.rope.freq_base": (
        mx.array(config.get("rope_theta", 10000), dtype=mx.float32)
    ),
}
rope_scaling = config.get("rope_scaling")

if rope_scaling is not None and (typ := rope_scaling.get("type")):
    rope_factor = rope_scaling.get("factor")
    f_rope_scale = rope_factor
    if typ == "linear":
        rope_scaling_type = "linear"
        metadata["llama.rope.scaling.type"] = rope_scaling_type
        metadata["llama.rope.scaling.factor"] = mx.array(f_rope_scale)

# add quantization metadata
quantization = config.get("quantization", None)
metadata["general.file_type"] = mx.array(
    (GGMLFileType.GGML_TYPE_Q4_0 if quantization else GGMLFileType.GGML_TYPE_F16).value,
    dtype=mx.uint32,
)
metadata["general.quantization_version"] = mx.array(
    (GGMLFileType.GGML_TYPE_Q4_0 if quantization else GGMLFileType.GGML_TYPE_F16).value,
    dtype=mx.uint32,
)

metadata["general.name"] = "mistralai/Mistral-7B-v0.1"
metadata["general.architecture"] = "llama"
metadata["general.alignment"] = mx.array(32, dtype=mx.uint32)
# add metadata for vocab
metadata["tokenizer.ggml.model"] = "llama"
tokens = []
scores = []
toktypes = []

for text, score, toktype in vocab.all_tokens():
    tokens.append(text)
    scores.append(score)
    toktypes.append(toktype.value)

assert len(tokens) == vocab.vocab_size

metadata["tokenizer.ggml.tokens"] = tokens
metadata["tokenizer.ggml.scores"] = mx.array(scores, dtype=mx.float32)
metadata["tokenizer.ggml.token_type"] = mx.array(toktypes, dtype=mx.uint32)

metadata["tokenizer.ggml.bos_token_id"] = mx.array(
    vocab.sentencepiece_tokenizer.bos_id(), dtype=mx.uint32
)
metadata["tokenizer.ggml.eos_token_id"] = mx.array(
    vocab.sentencepiece_tokenizer.eos_id(), dtype=mx.uint32
)
metadata["tokenizer.ggml.unknown_token_id"] = mx.array(
    vocab.sentencepiece_tokenizer.unk_id(), dtype=mx.uint32
)

metadata = {k: v for k, v in metadata.items() if v is not None}

weights = {
    k: v.astype(mx.float32).astype(mx.float16) if v.dtype == mx.bfloat16 else v
    for k, v in weights.items()
}

mx.save_gguf("mlx_model.gguf", weights, metadata)
