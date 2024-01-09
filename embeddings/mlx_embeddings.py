# Based on the MLX BERT implementation from mlx-examples, but
# designed to easily pull any HuggingFace BERT embeddings model
# and use it for batch inference for text embeddings.
# It also implements the proper API to be evaluated in MTEB
# and used like a sentence transformer model.
import torch
from typing import Literal
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten, tree_map
from dataclasses import dataclass
from typing import Optional
import numpy as np
from transformers import BertTokenizer, BertConfig, AutoModel

def replace_key(key: str) -> str:
    key = key.replace(".layer.", ".layers.")
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".ln1.")
    key = key.replace(".output.LayerNorm.", ".ln2.")
    key = key.replace(".intermediate.dense.", ".linear1.")
    key = key.replace(".output.dense.", ".linear2.")
    key = key.replace(".LayerNorm.", ".norm.")
    key = key.replace("pooler.dense.", "pooler.")
    return key

class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = nn.MultiHeadAttention(config.hidden_size, config.num_attention_heads, bias=True)
        # self.attention = BertAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x

class TransformerEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(config)
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
    ) -> tuple[mx.array, mx.array]:
        if token_type_ids is None:
            # just make it all 0s
            token_type_ids = mx.zeros_like(input_ids)
        x = self.embeddings(input_ids, token_type_ids)

        if attention_mask is not None:
            # convert 0's to -infs, 1's to 0's, and make it broadcastable
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        y = self.encoder(x, attention_mask) # shape: B, L, D
        return y, mx.tanh(self.pooler(y[:, 0])) # shape: B, D
    
    @classmethod
    def from_hugging_face(cls, model_path: str, precision_nbits: int = 8):
        if precision_nbits not in [2, 4, 8, 32]:
            raise ValueError("precision_nbits must be one of 2, 4, 8, 32")
        config = BertConfig.from_pretrained(model_path)
        torch_weights = AutoModel.from_pretrained(model_path).state_dict()

        # figure out how to convert torch weights to mx weights
        mx_weights = {
            replace_key(key): mx.array(tensor.numpy()) for key, tensor in torch_weights.items()
        }
        mlx_model = cls(config)
        mlx_model.update(
            tree_unflatten(list(mx_weights.items()))
        )
        if precision_nbits == 32:
            print("Keeping in fp32 precision")
        else:
            print(f"Quantizing to {precision_nbits} bits")
            nn.QuantizedLinear.quantize_module(mlx_model, bits=precision_nbits)
        tokenizer = BertTokenizer.from_pretrained(model_path)

        return mlx_model, tokenizer


class MLXEmbeddingModel:
    def __init__(
        self,
        model_path: str, # path to model on huggingface. must be BERT
        max_length: int = 512,
        pooling_strategy: Literal["mean", "cls"] = "cls",
        normalize: bool = True,
        precision_nbits: int = 8
    ):
        super().__init__()
        self.model, self.tokenizer = Bert.from_hugging_face(model_path, precision_nbits)
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
    
    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        tokenized = self.tokenizer(
            sentences,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        hidden_states, pooler_output = self.model(
            input_ids=mx.array(tokenized["input_ids"]),
            token_type_ids=mx.array(tokenized["token_type_ids"]),
            attention_mask=mx.array(tokenized["attention_mask"]),
        )
        hidden_states, pooler_output = np.array(hidden_states), np.array(pooler_output)
        attn_mask = np.array(tokenized["attention_mask"])
        if self.pooling_strategy == "mean":
            pooled = np.sum(hidden_states * np.expand_dims(attn_mask, -1), axis=1) / np.sum(attn_mask, axis=1, keepdims=True)
        else:
            pooled = pooler_output
        if self.normalize:
            pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled
    
def test_embeddings():
    mx_model, tokenizer = Bert.from_hugging_face("BAAI/bge-small-en-v1.5", precision_nbits=16)
    torch_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
    torch_model.eval()
    batch = [
        "This is an example of BERT working on MLX.",
        "A second string",
        "This is another string.",
    ]
    mx_tokens = tokenizer(batch, return_tensors="np", padding=True)
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    mx_embed_out = mx_model.embeddings(**{
        key: mx.array(value) for key, value in mx_tokens.items() if key in ["input_ids", "token_type_ids"]
    })
    
    torch_embed_out = torch_model.embeddings(**{
        key: value for key, value in torch_tokens.items() if key in ["input_ids", "token_type_ids"]
    })
    
    assert np.allclose(np.array(mx_embed_out), torch_embed_out.detach().numpy(), atol=1e-2), "Embeddings mismatch"