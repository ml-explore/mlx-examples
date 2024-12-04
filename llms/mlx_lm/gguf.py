import importlib
import re
import tempfile
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple, Union

import gguf
import mlx.core as mx
import mlx.nn as nn
from gguf import GGMLQuantizationType
from gguf.gguf_reader import GGUFReader
from transformers import AutoTokenizer

from .tokenizer_utils import TokenizerWrapper


class TokenType(IntEnum):
    NORMAL = 1
    UNKNOWN = 2
    CONTROL = 3
    USER_DEFINED = 4
    UNUSED = 5
    BYTE = 6


class GGMLFileType(IntEnum):
    GGML_TYPE_F16 = 1


# copied from https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L455
class HfVocab:
    def __init__(
        self,
        fname_tokenizer: Path,
        fname_added_tokens: Optional[Union[Path, None]] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            fname_tokenizer,
            cache_dir=fname_tokenizer,
            local_files_only=True,
        )
        self.added_tokens_list = []
        self.added_tokens_dict = dict()
        self.added_tokens_ids = set()
        for tok, tokidx in sorted(
            self.tokenizer.get_added_vocab().items(), key=lambda x: x[1]
        ):
            if tokidx >= self.tokenizer.vocab_size:
                self.added_tokens_list.append(tok)
                self.added_tokens_dict[tok] = tokidx
                self.added_tokens_ids.add(tokidx)
        self.specials = {
            tok: self.tokenizer.get_vocab()[tok]
            for tok in self.tokenizer.all_special_tokens
        }
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.vocab_size_base = self.tokenizer.vocab_size
        self.vocab_size = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer = fname_tokenizer
        self.fname_added_tokens = fname_added_tokens

    def hf_tokens(self) -> Iterable[Tuple[bytes, float, TokenType]]:
        reverse_vocab = {
            id: encoded_tok for encoded_tok, id in self.tokenizer.get_vocab().items()
        }
        for token_id in range(self.vocab_size_base):
            if token_id in self.added_tokens_ids:
                continue
            token_text = reverse_vocab[token_id]
            yield token_text, self.get_token_score(token_id), self.get_token_type(
                token_id, token_text, self.special_ids
            )

    def get_token_type(
        self, token_id: int, token_text: bytes, special_ids: Set[int]
    ) -> TokenType:
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", token_text):
            return TokenType.BYTE
        return TokenType.CONTROL if token_id in special_ids else TokenType.NORMAL

    def get_token_score(self, token_id: int) -> float:
        return -1000.0

    def added_tokens(self) -> Iterable[Tuple[bytes, float, TokenType]]:
        for text in self.added_tokens_list:
            if text in self.specials:
                toktype = self.get_token_type(self.specials[text], "", self.special_ids)
                score = self.get_token_score(self.specials[text])
            else:
                toktype = TokenType.USER_DEFINED
                score = -1000.0
            yield text, score, toktype

    def has_newline_token(self):
        return "<0x0A>" in self.tokenizer.vocab or "\n" in self.tokenizer.vocab

    def all_tokens(self) -> Iterable[Tuple[bytes, float, TokenType]]:
        yield from self.hf_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<HfVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"

    @staticmethod
    def load(path: Path) -> "HfVocab":
        added_tokens_path = path.parent / "added_tokens.json"
        return HfVocab(path, added_tokens_path if added_tokens_path.exists() else None)


def translate_weight_names(name):
    name = name.replace("model.layers.", "blk.")
    # for mixtral gate
    name = name.replace("block_sparse_moe.gate", "ffn_gate_inp")
    # for mixtral experts ffns
    pattern = r"block_sparse_moe\.experts\.(\d+)\.w1\.weight"
    replacement = r"ffn_gate.\1.weight"
    name = re.sub(pattern, replacement, name)
    pattern = r"block_sparse_moe\.experts\.(\d+)\.w2\.weight"
    replacement = r"ffn_down.\1.weight"
    name = re.sub(pattern, replacement, name)
    pattern = r"block_sparse_moe\.experts\.(\d+)\.w3\.weight"
    replacement = r"ffn_up.\1.weight"
    name = re.sub(pattern, replacement, name)

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


def permute_weights(weights, n_head, n_head_kv=None):
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    reshaped = weights.reshape(
        n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:]
    )
    swapped = reshaped.swapaxes(1, 2)
    final_shape = weights.shape
    return swapped.reshape(final_shape)


def prepare_metadata(config, vocab):
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
            if config.get("rms_norm_eps") is not None
            else None
        ),
        "llama.rope.freq_base": (
            mx.array(config.get("rope_theta", 10000), dtype=mx.float32)
            if config.get("rope_theta") is not None
            else None
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

    metadata["general.file_type"] = mx.array(
        GGMLFileType.GGML_TYPE_F16.value,
        dtype=mx.uint32,
    )
    metadata["general.quantization_version"] = mx.array(
        GGMLFileType.GGML_TYPE_F16.value,
        dtype=mx.uint32,
    )
    metadata["general.name"] = config.get("_name_or_path", "llama").split("/")[-1]
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
    if vocab.tokenizer.bos_token_id is not None:
        metadata["tokenizer.ggml.bos_token_id"] = mx.array(
            vocab.tokenizer.bos_token_id, dtype=mx.uint32
        )
    if vocab.tokenizer.eos_token_id is not None:
        metadata["tokenizer.ggml.eos_token_id"] = mx.array(
            vocab.tokenizer.eos_token_id, dtype=mx.uint32
        )
    if vocab.tokenizer.unk_token_id is not None:
        metadata["tokenizer.ggml.unknown_token_id"] = mx.array(
            vocab.tokenizer.unk_token_id, dtype=mx.uint32
        )

    metadata = {k: v for k, v in metadata.items() if v is not None}
    return metadata


def convert_to_gguf(
    model_path: Union[str, Path],
    weights: dict,
    config: dict,
    output_file_path: str,
):
    if isinstance(model_path, str):
        model_path = Path(model_path)

    quantization = config.get("quantization", None)
    if quantization:
        raise NotImplementedError(
            "Conversion of quantized models is not yet supported."
        )
    print("Converting to GGUF format")
    # https://github.com/ggerganov/llama.cpp/blob/master/convert.py#L1182 seems relate to llama.cpp's multihead attention
    weights = {
        k: (
            permute_weights(
                v, config["num_attention_heads"], config["num_attention_heads"]
            )
            if "self_attn.q_proj.weight" in k
            else (
                permute_weights(
                    v, config["num_attention_heads"], config["num_key_value_heads"]
                )
                if "self_attn.k_proj.weight" in k
                else v
            )
        )
        for k, v in weights.items()
    }

    # rename weights for gguf format
    weights = {translate_weight_names(k): v for k, v in weights.items()}

    if not (model_path / "tokenizer.json").exists():
        raise ValueError("Tokenizer json not found")

    vocab = HfVocab.load(model_path)
    metadata = prepare_metadata(config, vocab)

    weights = {
        k: (
            v.astype(mx.float32).astype(mx.float16)
            if v.dtype == mx.bfloat16
            else v.astype(mx.float32) if "norm" in k else v
        )
        for k, v in weights.items()
    }

    output_file_path = output_file_path
    mx.save_gguf(output_file_path, weights, metadata)
    print(f"Converted GGUF model saved as: {output_file_path}")


# Adapted from https://github.com/antirez/gguf-tools/blob/4e6455ecaf92b1a59e6a3291646459af3154bef5/gguflib.c#L568
def parse_q4_k(tensor):
    bits = 4
    pack_factor = 32 // bits
    group_size = 32
    block_size = 144

    data = mx.array(tensor.data)
    shape = [int(d) for d in reversed(tensor.shape)]
    wshape = (*shape[:-1], shape[-1] // pack_factor)
    gshape = (*shape[:-1], shape[-1] // group_size)
    num_blocks = data.size // block_size
    kernel = mx.fast.metal_kernel(
        name="parse_q4_k",
        input_names=["data"],
        output_names=["w", "scales", "biases"],
        header="""
        typedef struct {
            float16_t d;
            float16_t d_min;
            uint8_t scales[12];
            uint8_t qs[128];
        } block_q4_K;
        """,
        source="""
        uint elem = thread_position_in_grid.x;

        const device block_q4_K* block = reinterpret_cast<const device block_q4_K*>(data);

        block += elem;
        w += elem * 32;
        scales += elem * 8;
        biases += elem * 8;

        // First unpack the quantized scales/biases
        for (int j = 0; j < 8; j++) {
        uint8_t d, m;
        if (j < 4) {
            d = block->scales[j] & 63;
            m = block->scales[j + 4] & 63;
        } else {
            d = (block->scales[j + 4] & 0xF) | ((block->scales[j - 4] >> 6) << 4);
            m = (block->scales[j + 4] >> 4) | ((block->scales[j - 0] >> 6) << 4);
        }
        scales[j] = d * block->d;
        biases[j] = -m * block->d_min;
        }

        uint32_t outputs[32] = {0};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 32; j++) {
                uint8_t val = block->qs[i * 32 + j] & 0xf;
                int index = i * 8 + (j / 8);
                outputs[index] += val << (4 * (j % 8));
            }
            for (int j = 0; j < 32; j++) {
                uint8_t val = block->qs[i * 32 + j] >> 4;
                int index = i * 8 + 4 + (j / 8);
                outputs[index] += val << (4 * (j % 8));
            }
        }

        for (int i = 0; i < 32; i++) {
            w[i] = outputs[i];
        }
        """,
    )
    w, scales, biases = kernel(
        inputs=[data],
        grid=(num_blocks, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[wshape, gshape, gshape],
        output_dtypes=[mx.uint32, mx.float16, mx.float16],
    )
    return w, scales, biases


# Adapted from https://github.com/antirez/gguf-tools/blob/4e6455ecaf92b1a59e6a3291646459af3154bef5/gguflib.c#L658
def parse_q6_k(tensor):
    bits = 6
    group_size = 16
    block_size = 210

    data = mx.array(tensor.data)
    shape = [int(d) for d in reversed(tensor.shape)]
    wshape = (*shape[:-1], shape[-1] * bits // 8)
    gshape = (*shape[:-1], shape[-1] // group_size)
    num_blocks = data.size // block_size
    kernel = mx.fast.metal_kernel(
        name="parse_q6_k",
        input_names=["data"],
        output_names=["w", "scales", "biases"],
        header="""
        typedef struct {
            uint8_t ql[128];      // quants, lower 4 bits
            uint8_t qh[64];      // quants, upper 2 bits
            int8_t  scales[16]; // scales, quantized with 8 bits
            float16_t d;             // super-block scale
        } block_q6_K;
        """,
        source="""
        uint elem = thread_position_in_grid.x;

        const device block_q6_K* block = reinterpret_cast<const device block_q6_K*>(data);

        block += elem;
        w += elem * 192;
        scales += elem * 16;
        biases += elem * 16;

        const device uint8_t* ql = &block->ql[0];
        const device uint8_t* qh = &block->qh[0];
        const device int8_t* bscales = &block->scales[0];

        uint32_t output = 0;
        for (int cluster = 0; cluster < 2; cluster++) {
            for (uint64_t j = 0; j < 128; j++) {
                uint8_t val = ((ql[j%64] >> (j/64*4)) & 0xF) | (((qh[j%32] >> (j/32*2)) & 3) << 4);

                output += val << (6 * (j % 4));

                // Every 4 values write out 3 bytes
                if (j % 4 == 3) {
                    w[0] = output & 0xff;
                    w[1] = (output & 0xff00) >> 8;
                    w[2] = (output & 0xff0000) >> 16;
                    w += 3;
                    output = 0;
                }

                if (j % 16 == 0) {
                    scales[j/16] = block->d * bscales[j/16];
                    biases[j/16] = -32.0f * scales[j/16];
                }
            }
            ql += 64;
            qh += 32;
            bscales += 8;
            scales += 8;
            biases += 8;
        }
        """,
    )
    w, scales, biases = kernel(
        inputs=[data],
        grid=(num_blocks, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[wshape, gshape, gshape],
        output_dtypes=[mx.uint8, mx.float16, mx.float16],
    )
    w = mx.view(w, dtype=mx.uint32)
    return w, scales, biases


def parse_gguf_tensor(tensor):
    from gguf import GGMLQuantizationType

    if tensor.tensor_type == GGMLQuantizationType.Q4_K:
        return parse_q4_k(tensor)
    elif tensor.tensor_type == GGMLQuantizationType.Q6_K:
        return parse_q6_k(tensor)
    elif tensor.tensor_type in [GGMLQuantizationType.F16, GGMLQuantizationType.F32]:
        return mx.array(tensor.data)
    else:
        raise NotImplementedError(f"Type: {tensor.tensor_type} is not yet supported.")


def convert_name(name):
    name = name.replace("blk", "model.layers")
    name = name.replace("attn_norm", "input_layernorm")
    name = name.replace("ffn_norm", "post_attention_layernorm")
    name = name.replace("attn_q", "self_attn.q_proj")
    name = name.replace("attn_k", "self_attn.k_proj")
    name = name.replace("attn_v", "self_attn.v_proj")
    name = name.replace("attn_output", "self_attn.o_proj")
    name = name.replace("ffn_up", "mlp.up_proj")
    name = name.replace("ffn_down", "mlp.down_proj")
    name = name.replace("ffn_gate", "mlp.gate_proj")
    if "output_norm" in name:
        name = name.replace("output_norm", "model.norm")
    else:
        name = name.replace("output", "lm_head")
    name = name.replace("token_embd", "model.embed_tokens")
    return name


FIELD_MAPPING = {
    "{model}.embedding_length": "hidden_size",
    "{model}.feed_forward_length": "intermediate_size",
    "{model}.attention.head_count": "num_attention_heads",
    "{model}.attention.head_count_kv": "num_key_value_heads",
    "{model}.block_count": "num_hidden_layers",
    "{model}.attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "{model}.rope.freq_base": "rope_theta",
}


QUANT_MAPPING = {
    GGMLQuantizationType.Q4_K: {
        "bits": 4,
        "group_size": 32,
    },
    GGMLQuantizationType.Q6_K: {
        "bits": 6,
        "group_size": 16,
    },
    GGMLQuantizationType.F16: None,
    GGMLQuantizationType.F32: None,
}


# from https://github.com/ggerganov/llama.cpp/blob/40c6d79fb52f995f47507fedfeaae2ac05d9b35c/gguf-py/scripts/gguf_new_metadata.py#L46
def decode_field(field):
    if field and field.types:
        main_type = field.types[0]

        if main_type == gguf.GGUFValueType.ARRAY:
            sub_type = field.types[-1]

            if sub_type == gguf.GGUFValueType.STRING:
                return [
                    str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data
                ]
            else:
                return [pv for idx in field.data for pv in field.parts[idx].tolist()]
        if main_type == gguf.GGUFValueType.STRING:
            return str(bytes(field.parts[-1]), encoding="utf-8")
        else:
            return field.parts[-1][0]

    return None


def load_gguf(model_path: str) -> tuple[nn.Module, TokenizerWrapper]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_name = Path(model_path).name
        (Path(tmp_dir) / base_name).symlink_to(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tmp_dir, gguf_file=base_name)

    reader = GGUFReader(model_path)
    model_type = "qwen2"
    config = {
        "model_type": model_type,
        "vocab_size": tokenizer.vocab_size,
        "tie_word_embeddings": False,
    }
    mapping = {k.format(model=model_type): v for k, v in FIELD_MAPPING.items()}
    for field in reader.fields:
        if field in mapping:
            config[mapping[field]] = decode_field(reader.get_field(field))
    config["quantization"] = {}

    weights = {}

    # Look for any extra gguf files
    parts = Path(model_path).name.split("-")
    parts[-3] = "*"
    gguf_pattern = "-".join(parts)

    for filename in Path(model_path).parent.glob(gguf_pattern):
        reader = GGUFReader(str(filename))
        for tensor in reader.tensors:
            w = parse_gguf_tensor(tensor)
            mx.eval(w)
            name = convert_name(tensor.name)
            base_name = ".".join(name.split(".")[:-1])
            if quant := QUANT_MAPPING[tensor.tensor_type]:
                config["quantization"][base_name] = quant
            if len(w) == 3:
                w, scales, biases = w
                weights[name] = w
                weights[base_name + ".scales"] = scales
                weights[base_name + ".biases"] = biases
            else:
                weights[name] = w

    arch = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    model_class, model_args_class = arch.Model, arch.ModelArgs

    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    quant_config = config["quantization"]

    def pred(p, m):
        return quant_config.get(p)

    nn.quantize(model, class_predicate=pred)
    model.load_weights(list(weights.items()))

    model.eval()
    return model, tokenizer
