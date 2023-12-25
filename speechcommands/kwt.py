from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

__all__ = ["KWT", "kwt1", "kwt2", "kwt3"]


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def __call__(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, n, 3, h, -1).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(attn, axis=-1)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)
        x = self.out(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

    def __call__(self, x):
        x = self.norm1(self.attn(x)) + x
        x = self.norm2(self.ff(x)) + x
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = []
        for _ in range(depth):
            self.layers.append(Block(dim, heads, mlp_dim, dropout=dropout))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class KWT(nn.Module):
    """
    Implements the Keyword Transformer (KWT) [1] model.

    KWT is essentially a vision transformer [2] with minor modifications:
    - Instead of square patches, KWT uses rectangular patches -> a patch
      across frequency for every timestep
    - KWT modules apply layer normalization after attention/feedforward layers

    [1] https://arxiv.org/abs/2104.11178
    [2] https://arxiv.org/abs/2010.11929

    Parameters
    ----------
    input_res: tuple of ints
        Input resolution (time, frequency)
    patch_res: tuple of ints
        Patch resolution (time, frequency)
    num_classes: int
        Number of classes
    dim: int
        Model Embedding dimension
    depth: int
        Number of transformer layers
    heads: int
        Number of attention heads
    mlp_dim: int
        Feedforward hidden dimension
    pool: str
        Pooling type, either "cls" or "mean"
    in_channels: int, optional
        Number of input channels
    dropout: float, optional
        Dropout rate
    emb_dropout: float, optional
        Embedding dropout rate
    """

    def __init__(
        self,
        input_res,
        patch_res,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="mean",
        in_channels=1,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_patches = int(
            (input_res[0] / patch_res[0]) * (input_res[1] / patch_res[1])
        )
        self.dim = dim

        self.patch_embedding = nn.Conv2d(
            in_channels, dim, kernel_size=patch_res, stride=patch_res
        )
        self.pos_embedding = mx.random.truncated_normal(
            -0.01,
            0.01,
            (self.num_patches + 1, dim),
        )
        self.cls_token = mx.random.truncated_normal(-0.01, 0.01, (dim,))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.pool = pool
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def num_params(self):
        nparams = sum(x.size for k, x in tree_flatten(self.parameters()))
        return nparams

    def __call__(self, x):
        if x.ndim != 4:
            x = mx.expand_dims(x, axis=-1)
        x = self.patch_embedding(x)
        x = x.reshape(x.shape[0], -1, self.dim)
        assert x.shape[1] == self.num_patches

        cls_tokens = mx.broadcast_to(self.cls_token, (x.shape[0], 1, self.dim))
        x = mx.concatenate((cls_tokens, x), axis=1)

        x = x + self.pos_embedding

        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(axis=1) if self.pool == "mean" else x[:, 0]
        x = self.mlp_head(x)
        return x


def parse_kwt_args(**kwargs):
    input_res = kwargs.pop("input_res", [98, 40])
    patch_res = kwargs.pop("patch_res", [1, 40])
    num_classes = kwargs.pop("num_classes", 35)
    emb_dropout = kwargs.pop("emb_dropout", 0.1)
    return input_res, patch_res, num_classes, emb_dropout, kwargs


def kwt1(**kwargs):
    input_res, patch_res, num_classes, emb_dropout, kwargs = parse_kwt_args(**kwargs)
    return KWT(
        input_res,
        patch_res,
        num_classes,
        dim=64,
        depth=12,
        heads=1,
        mlp_dim=256,
        emb_dropout=emb_dropout,
        **kwargs
    )


def kwt2(**kwargs):
    input_res, patch_res, num_classes, emb_dropout, kwargs = parse_kwt_args(**kwargs)
    return KWT(
        input_res,
        patch_res,
        num_classes,
        dim=128,
        depth=12,
        heads=2,
        mlp_dim=512,
        emb_dropout=emb_dropout,
        **kwargs
    )


def kwt3(**kwargs):
    input_res, patch_res, num_classes, emb_dropout, kwargs = parse_kwt_args(**kwargs)
    return KWT(
        input_res,
        patch_res,
        num_classes,
        dim=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        emb_dropout=emb_dropout,
        **kwargs
    )
