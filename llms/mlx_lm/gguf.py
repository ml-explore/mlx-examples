import glob
import json
from enum import IntEnum
from pathlib import Path
from typing import Iterable

import mlx.core as mlx
from sentencepiece import SentencePieceProcessor
from utils import get_model_path


class TokenType(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


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


# TODO add metadata for vocab https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L1090-L1092


# load model and vocab
# load config
# check if quantized
# check if 4 bit quant, if not raise error for only support 4 bit quant for now
# add metadata for arch
# if not quant, write metadata to fp16
# if quant, write metadata to 4 bit quant
# write metadata for vocab

weights = load_weights("mlx-community/Mistral-7B-v0.1-hf-4bit-mlx")
config = json.load(open("mlx-community/Mistral-7B-v0.1-hf-4bit-mlx/config.json"))
vocab = SentencePieceVocab.load(
    "mlx-community/Mistral-7B-v0.1-hf-4bit-mlx/tokenizer.json"
)

print(weights.keys())
print(config)
print(vocab.all_tokens()[:10])
