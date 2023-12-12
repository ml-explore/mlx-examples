import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Optional, Tuple, List
from sentencepiece import SentencePieceProcessor

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map, tree_unflatten

@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int

    max_seq_len: int
    rope_theta: float

    num_experts: int
    num_experts_per_token: int
    gate_softmax: bool

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class MoEFeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.experts = [FeedForward(args) for _ in range(args.num_experts)]
        self.gate = nn.Linear(args.dim, args.num_experts, bias=False)
        self.num_experts_per_token = args.num_experts_per_token
        self.gate_softmax = args.gate_softmax
        print("Softmax gate:", self.gate_softmax)

    def __call__(self, x) -> mx.array:
        input_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        if self.gate_softmax:
            scores = mx.softmax(self.gate(x), axis=-1)
        else:
            scores = self.gate(x)

        expert_we

        return self.w2(nn.silu(self.w1(x)) * self.w3(x))