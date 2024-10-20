

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

@dataclass
class Mamba2Config:
    d_model: int # D
    n_layers: int
    d_head: int # todo : plutot n_heads non ?
    d_state: int = 64 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4
    n_groups: int = 1# todo : ??
    
    A_init_range: tuple = (1, 16)
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    dt_limit: tuple = (0.0, float("inf"))
    conv_init = None

    learnable_init_states: bool = False
    activation: str = "swish" # "swish" or "silu"
    
    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    chunk_size: int = 256
    use_mem_eff_path: bool = True
    dtype=None
    device=None

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments
        self.n_heads = self.d_inner // self.d_head
        assert self.d_inner % self.d_head == 0

        assert (self.d_inner / self.d_head) % 8 == 0, "requierement of causal_conv1d"

        # muP
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width

class Mamba2(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()

        self.config = config

        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x, caches=None):
        if caches is None:
            caches = [None] * self.config.n_layers

        for i, layer in enumerate(self.layers):
            x, caches[i] = layer(x, caches[i])

        if caches[0] == None:
            return x
        else:
            return x, caches

class ResidualBlock(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        
        self.config = config

        self.mixer = Mamba2Block(self.config)
        self.norm = RMSNorm(self.config.d_model, self.config.rms_norm_eps, self.config.mup)

    def forward(self, x, cache=None):
        output, cache = self.mixer(self.norm(x), cache)
        output = output + x
        return output, cache

class Mamba2Block(nn.Module):
    def __init__(self, config: Mamba2Config):
        super().__init__()
        factory_kwargs = {"device": config.device, "dtype": config.dtype}

        self.config = config        

        # [z, x, B, C, dt]
        d_in_proj = 2 * self.config.d_inner + 2 * self.config.n_groups * self.config.d_state + self.config.n_heads
        self.in_proj = nn.Linear(self.config.d_model, d_in_proj, bias=self.config.bias)

        conv_dim = self.config.d_inner + 2 * self.config.n_groups * self.config.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=self.config.conv_bias,
            kernel_size=self.config.d_conv,
            groups=conv_dim,
            padding=self.config.d_conv - 1,
            **factory_kwargs,
        )


        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.config.n_heads) * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        )
        dt = torch.clamp(dt, min=self.config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        assert self.config.A_init_range[0] > 0 and self.config.A_init_range[1] >= self.config.A_init_range[0]
        A = torch.empty(self.config.n_heads, dtype=torch.float32).uniform_(*self.config.A_init_range)
        self.A_log = torch.log(A).to(dtype=self.config.dtype)
        self.D = nn.Parameter(torch.ones(self.config.n_heads, device=self.config.device))

        self.norm = RMSNormGated(self.config.d_inner, eps=1e-5, norm_before_gate=False)

        self.out_proj = nn.Linear(self.config.d_inner, self.config.d_model, bias=self.config.bias)

    def forward(self, u, cache=None, seq_idx=None):
        """
        u: (B, L, D)
        Returns: out : same shape as u
        """

        batch, length, _ = u.shape

        return_cache = False
        if cache is not None and length > 1:
            cache = None
            return_cache = True
        
        if cache is not None:
            out, cache = self.step(u, cache)
            return out, cache

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.config.learnable_init_states else None
        dt_limit_kwargs = {} if self.config.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.config.dt_limit)

        z, xBC, dt = torch.split(
            zxbcdt, 
            [self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads],
            dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # 1D Convolution
        xBC = self.act(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)) # (B, L, self.d_inner + 2 * n_groups * d_state)


        x, B, C = torch.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state, self.config.n_groups * self.config.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.config.d_head),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.config.n_groups),
            rearrange(C, "b l (g n) -> b l g n", g=self.config.n_groups),
            chunk_size=self.config.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        # Multiply "gate" branch and apply extra normalization layer
        y = self.norm(y, z)
        out = self.out_proj(y)
        return out, cache
    
    def step(self, u, cache):
        """
        u: (B, 1, D)
        cache: (h_cache, conv_cache)
        """
        
        h_cache, conv_cache = cache

        zxbcdt = self.in_proj(u.squeeze(1))  # (B, 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.config.d_inner - 2 * self.config.n_groups * self.config.d_state - self.config.n_heads) // 2
        z0, x0, z, xBC, dt = torch.split(zxbcdt, [d_mlp, d_mlp, self.config.d_inner, self.config.d_inner + 2 * self.config.n_groups * self.config.d_state, self.config.n_heads], dim=-1)

        # conv step
        conv_cache.copy_(torch.roll(conv_cache, shifts=-1, dims=-1)) # update state (B, D, W)
        conv_cache[:, :, -1] = xBC
        xBC = torch.sum(conv_cache * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1) # (B, D)
        if self.conv1d.bias is not None:
            xBC = xBC + self.conv1d.bias
        xBC = self.act(xBC).to(dtype=x.dtype)
        
        x, B, C = torch.split(xBC, [self.config.d_inner, self.config.n_groups * self.config.d_state, self.config.n_groups * self.config.d_state], dim=-1)
        A = -torch.exp(self.A_log.float()) # (n_heads)

 
        A = repeat(A, "h -> h p n", p=self.config.d_head, n=self.config.d_state).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=self.config.d_head)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.config.d_head)
        D = repeat(self.D, "h -> h p", p=self.config.d_head)
        B = rearrange(B, "b (g n) -> b g n", g=self.config.n_groups)
        C = rearrange(C, "b (g n) -> b g n", g=self.config.n_groups)
        x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.config.d_head)
        
        y = selective_state_update(h_cache, x_reshaped, dt, A, B, C, D, z=None, dt_bias=dt_bias, dt_softplus=True)
        y = rearrange(y, "b h p -> b (h p)")
    
        #if self.rmsnorm:
        y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        return out.unsqueeze(1), (h_cache, conv_cache)

# taken straight from https://github.com/johnma2006/mamba-minimal/blob/master/model.py
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        self.use_mup = use_mup
        self.eps = eps

        # https://arxiv.org/abs/2404.05728, RMSNorm gains prevents muTransfer (section 4.2.3)
        if not use_mup:
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        if not self.use_mup:
            return output * self.weight
        else:
            return output