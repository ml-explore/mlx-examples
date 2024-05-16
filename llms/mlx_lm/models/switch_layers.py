import math

import mlx.core as mx
import mlx.nn as nn


class QuantizedSwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.silu,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()
        scale_in = math.sqrt(1 / input_dims)
        scale_hidden = math.sqrt(1 / hidden_dims)

        self.gate_proj_w, self.gate_proj_s, self.gate_proj_b = mx.quantize(
            mx.random.uniform(
                low=-scale_in,
                high=scale_in,
                shape=(num_experts, hidden_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
        )
        self.up_proj_w, self.up_proj_s, self.up_proj_b = mx.quantize(
            mx.random.uniform(
                low=-scale_in,
                high=scale_in,
                shape=(num_experts, hidden_dims, input_dims),
            ),
            group_size=group_size,
            bits=bits,
        )
        self.down_proj_w, self.down_proj_s, self.down_proj_b = mx.quantize(
            mx.random.uniform(
                low=-scale_hidden,
                high=scale_hidden,
                shape=(num_experts, input_dims, hidden_dims),
            ),
            group_size=group_size,
            bits=bits,
        )

        if bias:
            self.gate_proj_bias = mx.zeros((num_experts, hidden_dims))
            self.up_proj_bias = mx.zeros((num_experts, hidden_dims))
            self.down_proj_bias = mx.zeros((num_experts, input_dims))

        self.activation = activation

        self.num_experts = num_experts
        self.group_size = group_size
        self.bits = bits

    def __call__(self, x, indices) -> mx.array:
        have_bias = "up_proj_bias" in self

        # Prepare the input for block sparse matmuls
        x = mx.expand_dims(x, (-2, -3))

        # Up project the input
        x_up = mx.block_sparse_qmm(
            x,
            self["up_proj_w"],
            self["up_proj_s"],
            self["up_proj_b"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if have_bias:
            x_up = x_up + mx.expand_dims(self.up_proj_bias[indices], -2)

        # Compute the gate
        x_gate = mx.block_sparse_qmm(
            x,
            self["gate_proj_w"],
            self["gate_proj_s"],
            self["gate_proj_b"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if have_bias:
            x_gate = x_gate + mx.expand_dims(self.gate_proj_bias[indices], -2)

        # Compute the final projection
        x = self.activation(x_gate) * x_up
        x = mx.block_sparse_qmm(
            x,
            self["down_proj_w"],
            self["down_proj_s"],
            self["down_proj_b"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        x = x.squeeze(-2)
        if have_bias:
            x = x + self.down_proj_bias[indices]

        return x


class SwitchGLU(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.silu,
        bias: bool = False,
    ):
        super().__init__()
        scale_in = math.sqrt(1 / input_dims)
        scale_hidden = math.sqrt(1 / hidden_dims)

        self.gate_proj = mx.random.uniform(
            low=-scale_in,
            high=scale_in,
            shape=(num_experts, hidden_dims, input_dims),
        )
        self.up_proj = mx.random.uniform(
            low=-scale_in,
            high=scale_in,
            shape=(num_experts, hidden_dims, input_dims),
        )
        self.down_proj = mx.random.uniform(
            low=-scale_hidden,
            high=scale_hidden,
            shape=(num_experts, input_dims, hidden_dims),
        )

        if bias:
            self.gate_proj_bias = mx.zeros((num_experts, hidden_dims))
            self.up_proj_bias = mx.zeros((num_experts, hidden_dims))
            self.down_proj_bias = mx.zeros((num_experts, input_dims))

        self.activation = activation

        self.num_experts = num_experts

    def __call__(self, x, indices) -> mx.array:
        have_bias = "up_proj_bias" in self

        # Prepare the input for block sparse matmuls
        x = mx.expand_dims(x, (-2, -3))

        # Up project the input
        x_up = mx.block_sparse_mm(x, self.up_proj.swapaxes(-1, -2), rhs_indices=indices)
        if have_bias:
            x_up = x_up + mx.expand_dims(self.up_proj_bias[indices], -2)

        # Compute the gate
        x_gate = mx.block_sparse_mm(
            x, self.gate_proj.swapaxes(-1, -2), rhs_indices=indices
        )
        if have_bias:
            x_gate = x_gate + mx.expand_dims(self.gate_proj_bias[indices], -2)

        # Compute the final projection
        x = self.activation(x_gate) * x_up
        x = mx.block_sparse_mm(x, self.down_proj.swapaxes(-1, -2), rhs_indices=indices)
        x = x.squeeze(-2)
        if have_bias:
            x = x + self.down_proj_bias[indices]

        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        num_experts, hidden_dim, dim = self.gate_proj.shape

        qm = QuantizedSwitchGLU(dim, hidden_dim, num_experts)
        qm.gate_proj_w, qm.gate_proj_s, qm.gate_proj_b = mx.quantize(
            self.gate_proj, group_size=group_size, bits=bits
        )
        qm.up_proj_w, qm.up_proj_s, qm.up_proj_b = mx.quantize(
            self.up_proj, group_size=group_size, bits=bits
        )
        qm.down_proj_w, qm.down_proj_s, qm.down_proj_b = mx.quantize(
            self.down_proj, group_size=group_size, bits=bits
        )

        return qm

    def is_quantized(self, weights, prefix):
        return f"{prefix}.gate_proj_s" in weights


class QuantizedSwitchMLP(nn.Module):
    def __init__(
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.gelu_approx,
        bias: bool = False,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()
        scale_in = math.sqrt(1 / input_dims)
        scale_hidden = math.sqrt(1 / hidden_dims)

        self.fc1_w, self.fc1_s, self.fc1_b = mx.quantize(
            mx.random.uniform(
                low=-scale_in,
                high=scale_in,
                shape=(num_experts, hidden_dim, dim),
            ),
            group_size=group_size,
            bits=bits,
        )
        self.fc2_w, self.fc2_s, self.fc2_b = mx.quantize(
            mx.random.uniform(
                low=-scale_hidden,
                high=scale_hidden,
                shape=(num_experts, dim, hidden_dim),
            ),
            group_size=group_size,
            bits=bits,
        )
        if bias:
            self.fc1_bias = mx.zeros((num_experts, hidden_dims))
            self.fc2_bias = mx.zeros((num_experts, input_dims))

        self.act = activation

        self.num_experts = num_experts
        self.group_size = group_size
        self.bits = bits

    def __call__(self, x, indices) -> mx.array:
        have_bias = "fc1_bias" in self

        # Prepare the input for block sparse matmuls
        x = mx.expand_dims(x, (-2, -3))

        # Compute the MLP
        x = mx.block_sparse_qmm(
            x,
            self["fc1_w"],
            self["fc1_s"],
            self["fc1_b"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if have_bias:
            x = x + mx.expand_dims(self.fc1_bias[indices], -2)
        x = self.activation(x)
        x = mx.block_sparse_qmm(
            x,
            self["fc2_w"],
            self["fc2_s"],
            self["fc2_b"],
            rhs_indices=indices,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        x = x.squeeze(-2)
        if have_bias:
            x = x + self.fc2_bias[indices]

        return x


class SwitchMLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        activation=nn.gelu_approx,
        bias: bool = False,
    ):
        super().__init__()
        scale_in = math.sqrt(1 / input_dims)
        scale_hidden = math.sqrt(1 / hidden_dims)

        self.fc1 = mx.random.uniform(
            low=-scale_in,
            high=scale_in,
            shape=(num_experts, hidden_dims, input_dims),
        )
        self.fc2 = mx.random.uniform(
            low=-scale_hidden,
            high=scale_hidden,
            shape=(num_experts, input_dims, hidden_dims),
        )

        if bias:
            self.fc1_bias = mx.zeros((num_experts, hidden_dims))
            self.fc2_bias = mx.zeros((num_experts, input_dims))

        self.activation = activation
        self.num_experts = num_experts

    def __call__(self, x, indices) -> mx.array:
        have_bias = "fc1_bias" in self

        # Prepare the input for block sparse matmuls
        x = mx.expand_dims(x, (-2, -3))

        # Compute the MLP
        x = mx.block_sparse_mm(x, self.fc1.swapaxes(-1, -2), rhs_indices=indices)
        if have_bias:
            x = x + mx.expand_dims(self.fc1_bias[indices], -2)
        x = self.activation(x)
        x = mx.block_sparse_mm(x, self.fc2.swapaxes(-1, -2), rhs_indices=indices)
        x = x.squeeze(-2)
        if have_bias:
            x = x + self.fc2_bias[indices]

        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        num_experts, hidden_dim, dim = self.fc1.shape

        qm = QuantizedSwitchMLP(dim, hidden_dim, num_experts)
        qm.fc1_w, qm.fc1_s, qm.fc1_b = mx.quantize(
            self.fc1, group_size=group_size, bits=bits
        )
        qm.fc2_w, qm.fc2_s, qm.fc2_b = mx.quantize(
            self.fc2, group_size=group_size, bits=bits
        )

        return qm

    def is_quantized(self, weights, prefix):
        return f"{prefix}.fc1_s" in weights
