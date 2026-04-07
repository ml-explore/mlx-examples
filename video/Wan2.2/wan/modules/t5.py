# MLX implementation for t5.py
import logging
import math
from typing import Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

from .tokenizers import HuggingfaceTokenizer

__all__ = [
    'T5Model',
    'T5Encoder',
    'T5Decoder',
    'T5EncoderModel',
]


def fp16_clamp(x):
    if x.dtype == mx.float16:
        # Use same clamping as PyTorch for consistency
        clamp = 65504.0  # max value for float16
        return mx.clip(x, -clamp, clamp)
    return x


class GELU(nn.Module):
    def __call__(self, x):
        return 0.5 * x * (1.0 + mx.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))


class T5LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x):
        # Match PyTorch's approach: convert to float32 for stability
        x_float = x.astype(mx.float32) if x.dtype == mx.float16 else x
        variance = mx.mean(mx.square(x_float), axis=-1, keepdims=True)
        x_norm = x_float * mx.rsqrt(variance + self.eps)
        # Convert back to original dtype
        if x.dtype == mx.float16:
            x_norm = x_norm.astype(mx.float16)
        return self.weight * x_norm


class T5Attention(nn.Module):
    def __init__(self, dim, dim_attn, num_heads, dropout=0.0):
        assert dim_attn % num_heads == 0
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        # layers
        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, context=None, mask=None, pos_bias=None):
        """
        x:          [B, L1, C].
        context:    [B, L2, C] or None.
        mask:       [B, L2] or [B, L1, L2] or None.
        """
        # check inputs
        context = x if context is None else context
        b, l1, _ = x.shape
        _, l2, _ = context.shape
        n, c = self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x).reshape(b, l1, n, c)
        k = self.k(context).reshape(b, l2, n, c)
        v = self.v(context).reshape(b, l2, n, c)

        # transpose for attention: [B, N, L, C]
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        # compute attention (T5 does not use scaling)
        attn = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))  # [B, N, L1, L2]
        
        # add position bias if provided
        if pos_bias is not None:
            attn = attn + pos_bias
            
        # apply mask
        if mask is not None:
            if mask.ndim == 2:
                # [B, L2] -> [B, 1, 1, L2]
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                # [B, L1, L2] -> [B, 1, L1, L2]
                mask = mask[:, None, :, :]
            # Use very negative value that works well with float16
            min_value = -65504.0 if attn.dtype == mx.float16 else -1e9
            attn = mx.where(mask == 0, min_value, attn)

        # softmax and apply attention
        attn = mx.softmax(attn.astype(mx.float32), axis=-1).astype(attn.dtype)
        attn = self.dropout(attn)
        
        # apply attention to values
        x = mx.matmul(attn, v)  # [B, N, L1, C]
        
        # transpose back and reshape
        x = mx.transpose(x, (0, 2, 1, 3))  # [B, L1, N, C]
        x = x.reshape(b, l1, -1)
        
        # output projection
        x = self.o(x)
        x = self.dropout(x)
        return x


class T5FeedForward(nn.Module):
    def __init__(self, dim, dim_ffn, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn

        # layers
        self.gate_proj = nn.Linear(dim, dim_ffn, bias=False)
        self.gate_act = GELU()
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        gate = self.gate_act(self.gate_proj(x))
        x = self.fc1(x) * gate
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class T5SelfAttention(nn.Module):
    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True)

    def __call__(self, x, mask=None, pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.ffn(self.norm2(x)))
        return x


class T5CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        self.norm1 = T5LayerNorm(dim)
        self.self_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm2 = T5LayerNorm(dim)
        self.cross_attn = T5Attention(dim, dim_attn, num_heads, dropout)
        self.norm3 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn, dropout)
        self.pos_embedding = None if shared_pos else T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False)

    def __call__(self,
                x,
                mask=None,
                encoder_states=None,
                encoder_mask=None,
                pos_bias=None):
        e = pos_bias if self.shared_pos else self.pos_embedding(
            x.shape[1], x.shape[1])
        x = fp16_clamp(x + self.self_attn(self.norm1(x), mask=mask, pos_bias=e))
        x = fp16_clamp(x + self.cross_attn(
            self.norm2(x), context=encoder_states, mask=encoder_mask))
        x = fp16_clamp(x + self.ffn(self.norm3(x)))
        return x


class T5RelativeEmbedding(nn.Module):
    def __init__(self, num_buckets, num_heads, bidirectional, max_dist=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist

        # layers
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def __call__(self, lq, lk):
        # Create relative position matrix
        positions_q = mx.arange(lq)[:, None]
        positions_k = mx.arange(lk)[None, :]
        rel_pos = positions_k - positions_q
        
        # Apply bucketing
        rel_pos = self._relative_position_bucket(rel_pos)
        
        # Get embeddings
        rel_pos_embeds = self.embedding(rel_pos)
        
        # Reshape to [1, N, Lq, Lk]
        rel_pos_embeds = mx.transpose(rel_pos_embeds, (2, 0, 1))
        rel_pos_embeds = mx.expand_dims(rel_pos_embeds, 0)
        
        return rel_pos_embeds

    def _relative_position_bucket(self, rel_pos):
        # preprocess
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = mx.array(rel_pos > 0, dtype=mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros_like(rel_pos, dtype=mx.int32)
            rel_pos = -mx.minimum(rel_pos, mx.zeros_like(rel_pos))

        # embeddings for small and large positions
        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact
        
        # For large positions, use log scale
        rel_pos_large = max_exact + (
            mx.log(mx.array(rel_pos, dtype=mx.float32) / max_exact) /
            math.log(self.max_dist / max_exact) *
            (num_buckets - max_exact)
        ).astype(mx.int32)
        
        rel_pos_large = mx.minimum(rel_pos_large, num_buckets - 1)
        
        # Combine small and large position buckets
        rel_buckets = rel_buckets + mx.where(is_small, rel_pos, rel_pos_large)
        
        return rel_buckets


class T5Encoder(nn.Module):
    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        if isinstance(vocab, nn.Embedding):
            self.token_embedding = vocab
        else:
            self.token_embedding = nn.Embedding(vocab, dim)
            
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=True) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = [
            T5SelfAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                            shared_pos, dropout) for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids, mask=None):
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.shape[1],
                               x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Decoder(nn.Module):
    def __init__(self,
                 vocab,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 num_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_buckets = num_buckets
        self.shared_pos = shared_pos

        # layers
        if isinstance(vocab, nn.Embedding):
            self.token_embedding = vocab
        else:
            self.token_embedding = nn.Embedding(vocab, dim)
            
        self.pos_embedding = T5RelativeEmbedding(
            num_buckets, num_heads, bidirectional=False) if shared_pos else None
        self.dropout = nn.Dropout(dropout)
        self.blocks = [
            T5CrossAttention(dim, dim_attn, dim_ffn, num_heads, num_buckets,
                             shared_pos, dropout) for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    def __call__(self, ids, mask=None, encoder_states=None, encoder_mask=None):
        b, s = ids.shape

        # causal mask
        if mask is None:
            mask = mx.tril(mx.ones((1, s, s)))
        elif mask.ndim == 2:
            # Expand mask properly
            mask = mx.tril(mx.expand_dims(mask, 1).broadcast_to((b, s, s)))

        # layers
        x = self.token_embedding(ids)
        x = self.dropout(x)
        e = self.pos_embedding(x.shape[1],
                               x.shape[1]) if self.shared_pos else None
        for block in self.blocks:
            x = block(x, mask, encoder_states, encoder_mask, pos_bias=e)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class T5Model(nn.Module):
    def __init__(self,
                 vocab_size,
                 dim,
                 dim_attn,
                 dim_ffn,
                 num_heads,
                 encoder_layers,
                 decoder_layers,
                 num_buckets,
                 shared_pos=True,
                 dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.dim_attn = dim_attn
        self.dim_ffn = dim_ffn
        self.num_heads = num_heads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.num_buckets = num_buckets

        # layers
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.encoder = T5Encoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, encoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.decoder = T5Decoder(self.token_embedding, dim, dim_attn, dim_ffn,
                                 num_heads, decoder_layers, num_buckets,
                                 shared_pos, dropout)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def __call__(self, encoder_ids, encoder_mask, decoder_ids, decoder_mask):
        x = self.encoder(encoder_ids, encoder_mask)
        x = self.decoder(decoder_ids, decoder_mask, x, encoder_mask)
        x = self.head(x)
        return x


def init_mlx_weights(module, key):
    """Initialize weights for T5 model components to match PyTorch initialization"""
    
    def normal(key, shape, std=1.0):
        return mx.random.normal(key, shape) * std
    
    if isinstance(module, T5LayerNorm):
        module.weight = mx.ones_like(module.weight)
    elif isinstance(module, nn.Embedding):
        key = mx.random.split(key, 1)[0]
        module.weight = normal(key, module.weight.shape, std=1.0)
    elif isinstance(module, T5FeedForward):
        # Match PyTorch initialization
        key1, key2, key3 = mx.random.split(key, 3)
        module.gate_proj.weight = normal(key1, module.gate_proj.weight.shape, 
                                        std=module.dim**-0.5)
        module.fc1.weight = normal(key2, module.fc1.weight.shape, 
                                  std=module.dim**-0.5)
        module.fc2.weight = normal(key3, module.fc2.weight.shape, 
                                  std=module.dim_ffn**-0.5)
    elif isinstance(module, T5Attention):
        # Match PyTorch initialization
        key1, key2, key3, key4 = random.split(key, 4)
        module.q.weight = normal(key1, module.q.weight.shape, 
                                std=(module.dim * module.dim_attn)**-0.5)
        module.k.weight = normal(key2, module.k.weight.shape, 
                                std=module.dim**-0.5)
        module.v.weight = normal(key3, module.v.weight.shape, 
                                std=module.dim**-0.5)
        module.o.weight = normal(key4, module.o.weight.shape, 
                                std=(module.num_heads * module.dim_attn)**-0.5)
    elif isinstance(module, T5RelativeEmbedding):
        key = mx.random.split(key, 1)[0]
        module.embedding.weight = normal(key, module.embedding.weight.shape,
                                        std=(2 * module.num_buckets * module.num_heads)**-0.5)
    elif isinstance(module, nn.Linear):
        # Generic linear layer initialization
        key = mx.random.split(key, 1)[0]
        fan_in = module.weight.shape[1]
        bound = 1.0 / math.sqrt(fan_in)
        module.weight = mx.random.uniform(key, module.weight.shape, -bound, bound)
    
    return module


def _t5(name,
        encoder_only=False,
        decoder_only=False,
        return_tokenizer=False,
        tokenizer_kwargs={},
        **kwargs):
    # sanity check
    assert not (encoder_only and decoder_only)

    # params
    if encoder_only:
        model_cls = T5Encoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('encoder_layers')
        _ = kwargs.pop('decoder_layers')
    elif decoder_only:
        model_cls = T5Decoder
        kwargs['vocab'] = kwargs.pop('vocab_size')
        kwargs['num_layers'] = kwargs.pop('decoder_layers')
        _ = kwargs.pop('encoder_layers')
    else:
        model_cls = T5Model

    # init model
    model = model_cls(**kwargs)
    
    # Initialize weights properly
    key = mx.random.key(0)
    model = init_mlx_weights(model, key)

    # init tokenizer
    if return_tokenizer:
        from .tokenizers import HuggingfaceTokenizer
        tokenizer = HuggingfaceTokenizer(f'google/{name}', **tokenizer_kwargs)
        return model, tokenizer
    else:
        return model


def umt5_xxl(**kwargs):
    cfg = dict(
        vocab_size=256384,
        dim=4096,
        dim_attn=4096,
        dim_ffn=10240,
        num_heads=64,
        encoder_layers=24,
        decoder_layers=24,
        num_buckets=32,
        shared_pos=False,
        dropout=0.0)
    cfg.update(**kwargs)
    return _t5('umt5-xxl', **cfg)


class T5EncoderModel:
    def __init__(
        self,
        text_len,
        checkpoint_path=None,
        tokenizer_path=None,
    ):
        self.text_len = text_len
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False)
        
        if checkpoint_path:
            logging.info(f'loading {checkpoint_path}')
            # Load weights - assuming MLX format checkpoint
            weights = mx.load(checkpoint_path)
            model.update(tree_unflatten(list(weights.items())))
        
        self.model = model
        
        # init tokenizer
        from .tokenizers import HuggingfaceTokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path if tokenizer_path else 'google/umt5-xxl', 
            seq_len=text_len, 
            clean='whitespace')

    def __call__(self, texts):
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        tokenizer_output = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True)
        
        # Handle different tokenizer output formats
        if isinstance(tokenizer_output, tuple):
            ids, mask = tokenizer_output
        else:
            # Assuming dict output with 'input_ids' and 'attention_mask'
            ids = tokenizer_output['input_ids']
            mask = tokenizer_output['attention_mask']
        
        # Convert to MLX arrays if not already
        if not isinstance(ids, mx.array):
            ids = mx.array(ids)
        if not isinstance(mask, mx.array):
            mask = mx.array(mask)
        
        # Get sequence lengths
        seq_lens = mx.sum(mask > 0, axis=1)
        
        # Run encoder
        context = self.model(ids, mask)
        
        # Return variable length outputs
        # Convert seq_lens to Python list for indexing
        if seq_lens.ndim == 0:  # Single value
            seq_lens_list = [seq_lens.item()]
        else:
            seq_lens_list = seq_lens.tolist()
        
        return [context[i, :int(seq_lens_list[i])] for i in range(len(texts))]


# Utility function to convert PyTorch checkpoint to MLX
def convert_pytorch_checkpoint(pytorch_path, mlx_path):
    """Convert PyTorch checkpoint to MLX format"""
    import torch
    
    # Load PyTorch checkpoint
    pytorch_state = torch.load(pytorch_path, map_location='cpu')
    
    # Convert to numpy then to MLX
    mlx_state = {}
    for key, value in pytorch_state.items():
        if isinstance(value, torch.Tensor):
            # Handle the key mapping if needed
            mlx_key = key
            # Convert tensor to MLX array
            mlx_state[mlx_key] = mx.array(value.numpy())
    
    # Save MLX checkpoint
    mx.save(mlx_path, mlx_state)
    
    return mlx_state