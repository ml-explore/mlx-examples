import torch.nn as nn
import torch 
from configuration_mamba import MambaConfig
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Optional , Union ,Tuple

# Dear contributors of the https://github.com/johnma2006/mamba-minimal/tree/master repository, special thanks to Albert Gu and Tri Dao for their articles. (https://arxiv.org/abs/2312.00752)


class MambaRMSNorm(nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
    

class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.config = config

        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            bias=config.conv_bias,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        A = repeat(torch.arange(1, config.d_state + 1), 'n -> d n', d=config.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)
        self.norm = MambaRMSNorm(config.d_model)

    def forward(self, x):
        (b, l, d) = x.shape
        x_copy = x # There was a separate class for residual, I deleted that part and added it here.
        x = self.norm(x)
        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.config.d_inner, self.config.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y) + x_copy

        return output

    
    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.config.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b d_in l n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b d_in l n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y
    

class MambaModel(MambaPreTrainedModel):
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([MambaBlock(config) for _ in range(config.n_layer)])
        self.norm_f = MambaRMSNorm(config.d_model)

    def forward(self, input_ids: torch.LongTensor = None):
        x = self.embedding(input_ids)
        all_hidden_states = list()
        for layer in self.layers:
            x = layer(x)
            all_hidden_states.append(x)
        return self.norm_f(x)
    

class MambaForCausalLM(MambaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MambaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.model.embedding.weight

    
    def forward(self, input_ids: torch.LongTensor = None):
        hidden_states = self.model(input_ids=input_ids)
        return self.lm_head(hidden_states)
