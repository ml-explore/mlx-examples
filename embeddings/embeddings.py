# Based on the MLX BERT implementation from mlx-examples, but
# designed to easily pull any HuggingFace BERT embeddings model
# and use it for batch inference for text embeddings.
# It also implements the proper API to be evaluated in MTEB
# and used like a sentence transformer model.
import tqdm
from typing import Literal
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten, tree_map
from dataclasses import dataclass
from typing import Optional
import numpy as np
from transformers import AutoTokenizer, BertConfig, AutoModel

def replace_key(key: str) -> str:
    key = key.replace(".self.key.", ".key_proj.")
    key = key.replace(".self.query.", ".query_proj.")
    key = key.replace(".self.value.", ".value_proj.")
    key = key.replace(".attention.output.dense.", ".attention.out_proj.")
    key = key.replace(".attention.output.LayerNorm.", ".attention_norm.")
    return key

class BertIntermediate(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.gelu = nn.GELU()

    def __call__(self, x: mx.array) -> mx.array:
        return self.gelu(self.dense(x))
    
class BertOutput(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = nn.MultiHeadAttention(
            config.hidden_size, 
            config.num_attention_heads, 
            bias=True
        )
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        attention_out = self.attention_norm(x + attention_out)
        intermediate_out = self.intermediate(attention_out)
        layer_out = self.output(intermediate_out, attention_out)
        return layer_out

class TransformerEncoder(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.layer = [
            TransformerEncoderLayer(config)
            for i in range(config.num_hidden_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return x

class BertEmbeddings(nn.Module):
    def __init__(self, config: BertConfig):
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array, token_type_ids: mx.array) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )
        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.LayerNorm(embeddings)

class BertPooler(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return mx.tanh(pooled_output)

class Bert(nn.Module):
    def __init__(self, config: BertConfig):
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = BertPooler(config)

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
        return y, self.pooler(y)
    
    @classmethod
    def from_pretrained(cls, model_path: str, precision_nbits: int = 8):
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
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        return mlx_model, tokenizer


class EmbeddingModel:
    def __init__(
        self,
        model_path: str, # path to model on huggingface. must be BERT
        max_length: int = 512,
        pooling_strategy: Literal["mean", "cls"] = "cls",
        normalize: bool = True,
        precision_nbits: int = 8
    ):
        super().__init__()
        self.model, self.tokenizer = Bert.from_pretrained(model_path, precision_nbits)
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        self.normalize = normalize
    
    def encode(self, sentences, batch_size=32, show_progress=True, **kwargs):
        """
        Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        result = None
        if show_progress:
            pbar = tqdm.tqdm(total=len(sentences))
        else:
            pbar = None
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            tokenized = self.tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            hidden_states, pooler_output = self.model(
                input_ids=mx.array(tokenized["input_ids"]),
                token_type_ids=mx.array(tokenized["token_type_ids"]) if "token_type_ids" in tokenized else None,
                attention_mask=mx.array(tokenized["attention_mask"]),
            )
            hidden_states, pooler_output = np.array(hidden_states), np.array(pooler_output)
            attn_mask = np.array(tokenized["attention_mask"])
            if self.pooling_strategy == "mean":
                pooled = np.sum(hidden_states * np.expand_dims(attn_mask, -1), axis=1) / np.sum(attn_mask, axis=1, keepdims=True)
            elif self.pooling_strategy == "cls":
                pooled = pooler_output
            elif self.pooling_strategy == "first":
                pooled = hidden_states[:, 0]
            if self.normalize:
                pooled = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)

            if result is None:
                result = pooled
            else:
                result = np.concatenate([result, pooled], axis=0)
            if pbar is not None:
                pbar.update(len(batch))

        return result

def test_embeddings():
    mx_model, tokenizer = Bert.from_pretrained(
        "BAAI/bge-small-en-v1.5", 
        precision_nbits=32
    )
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