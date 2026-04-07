# Copyright © 2024 Apple Inc.

"""
Native MLX implementation of Silero VAD (Voice Activity Detection).

This module provides a pure MLX implementation of the Silero VAD model,
converted from the original ONNX weights. The model detects speech segments
in audio to improve transcription quality by filtering silent regions.

Architecture:
    1. STFT: Converts audio to frequency domain via learned conv basis
    2. Encoder: 4-layer 1D CNN with ReLU activations
    3. Decoder: LSTM + 1D Conv for speech probability output
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from huggingface_hub import hf_hub_download


class SileroSTFT(nn.Module):
    """STFT layer using learned convolutional basis.

    Converts raw audio to magnitude spectrogram using a learned basis
    stored as convolution weights (not traditional FFT).
    """

    def __init__(self, n_fft: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4

    def __call__(self, x: mx.array) -> mx.array:
        """Apply STFT.

        Args:
            x: Audio tensor [batch, samples]

        Returns:
            Magnitude spectrogram [batch, frames, n_fft//2 + 1]
        """
        pad_amount = self.n_fft // 2
        x = mx.pad(x, [(0, 0), (pad_amount, pad_amount)])
        x = mx.expand_dims(x, axis=-1)
        x = mx.conv1d(x, self.forward_basis, stride=self.hop_length)

        n_freq = self.n_fft // 2 + 1
        real = x[:, :, :n_freq]
        imag = x[:, :, n_freq:]
        magnitude = mx.sqrt(real**2 + imag**2 + 1e-9)

        return magnitude


class ConvBlock(nn.Module):
    """1D Convolution block with ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.conv(x))


class SileroEncoder(nn.Module):
    """4-layer CNN encoder for Silero VAD."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.blocks = [
            ConvBlock(input_dim, 128, 3),
            ConvBlock(128, 64, 3),
            ConvBlock(64, 64, 3),
            ConvBlock(64, 128, 3),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.blocks:
            x = block(x)
        return x


class SileroLSTMDecoder(nn.Module):
    """LSTM-based decoder for speech probability prediction.

    Uses nn.LSTM with proper state handling for streaming inference.
    """

    def __init__(self, input_size: int = 128, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.out_conv = nn.Conv1d(in_channels=hidden_size, out_channels=1, kernel_size=1)

    def __call__(
        self,
        x: mx.array,
        h: Optional[mx.array] = None,
        c: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Decode features to speech probability.

        Args:
            x: Encoded features [batch, frames, 128]
            h: Hidden state [batch, hidden_size] or None
            c: Cell state [batch, hidden_size] or None

        Returns:
            Tuple of:
                - Speech probability [batch, 1]
                - New hidden state [batch, hidden_size]
                - New cell state [batch, hidden_size]
        """
        batch_size = x.shape[0]

        if h is None:
            h = mx.zeros((batch_size, self.hidden_size))
        if c is None:
            c = mx.zeros((batch_size, self.hidden_size))

        # MLX LSTM returns (h_states, c_states) both of shape [batch, seq, hidden]
        h_states, c_states = self.lstm(x, hidden=h, cell=c)

        # The last timestep gives us the new hidden and cell states
        h_new = h_states[:, -1, :]
        c_new = c_states[:, -1, :]

        # Apply ReLU to hidden states (which is also the output)
        lstm_out = nn.relu(h_states)

        # Conv and sigmoid for final probability
        out = self.out_conv(lstm_out)
        out = mx.sigmoid(out)

        # Return [batch, 1] probability from last timestep
        out = out[:, -1, :]

        return out, h_new, c_new


class SileroVADModel(nn.Module):
    """Complete Silero VAD model in native MLX."""

    def __init__(self, sample_rate: int = 16000):
        super().__init__()

        if sample_rate == 16000:
            n_fft = 256
            self.window_size = 512
            self.context_size = 64
        elif sample_rate == 8000:
            n_fft = 128
            self.window_size = 256
            self.context_size = 32
        else:
            raise ValueError(f"Unsupported sample rate: {sample_rate}")

        self.sample_rate = sample_rate
        n_freq = n_fft // 2 + 1

        self.stft = SileroSTFT(n_fft=n_fft)
        self.encoder = SileroEncoder(input_dim=n_freq)
        self.decoder = SileroLSTMDecoder(input_size=128, hidden_size=128)

    def __call__(
        self,
        audio: mx.array,
        h: Optional[mx.array] = None,
        c: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Process audio chunk and return speech probability.

        Args:
            audio: Audio chunk [batch, samples]
            h: Hidden state from previous call
            c: Cell state from previous call

        Returns:
            Tuple of (probability, h_new, c_new)
        """
        spec = self.stft(audio)
        features = self.encoder(spec)
        features = mx.mean(features, axis=1, keepdims=True)
        prob, h_new, c_new = self.decoder(features, h, c)

        return prob, h_new, c_new

    def reset_state(self, batch_size: int = 1) -> Tuple[mx.array, mx.array]:
        """Create fresh LSTM state."""
        h = mx.zeros((batch_size, self.decoder.hidden_size))
        c = mx.zeros((batch_size, self.decoder.hidden_size))
        return h, c


def _reorder_lstm_gates_onnx_to_mlx(weights: np.ndarray, hidden_size: int) -> np.ndarray:
    """Reorder LSTM gate weights from ONNX to MLX format.

    ONNX gate order: [i, o, f, c] (input, output, forget, cell)
    MLX gate order:  [i, f, g, o] (input, forget, gate/cell, output)

    Reordering: i->i, o->o, f->f, c->g
    ONNX indices: 0=i, 1=o, 2=f, 3=c
    MLX indices:  0=i, 1=f, 2=g, 3=o
    """
    # Split into 4 gates
    i = weights[:hidden_size]  # input gate
    o = weights[hidden_size : 2 * hidden_size]  # output gate
    f = weights[2 * hidden_size : 3 * hidden_size]  # forget gate
    c = weights[3 * hidden_size : 4 * hidden_size]  # cell gate
    # Reorder to MLX format: [i, f, g, o]
    return np.concatenate([i, f, c, o], axis=0)


def _convert_onnx_weights(sample_rate: int = 16000) -> dict:
    """Extract weights from ONNX model for given sample rate.

    The ONNX model has nested If nodes:
    - 16kHz: If_0 -> then_branch -> If_0 -> then_branch
    - 8kHz:  If_0 -> else_branch -> If_0 -> else_branch

    We recursively extract all weights with their full path prefixes.
    """
    import onnx
    from onnx import numpy_helper

    model_path = hf_hub_download(
        repo_id="onnx-community/silero-vad", filename="onnx/model.onnx"
    )
    model = onnx.load(model_path)

    branch_name = "then_branch" if sample_rate == 16000 else "else_branch"
    weights = {}

    def extract_all_recursive(graph, prefix=""):
        """Recursively extract all initializers from graph and subgraphs."""
        for init in graph.initializer:
            full_name = prefix + init.name
            arr = numpy_helper.to_array(init)
            weights[full_name] = arr

        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    sub_prefix = f"{prefix}{node.name}__{attr.name}__"
                    extract_all_recursive(attr.g, sub_prefix)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for i, g in enumerate(attr.graphs):
                        sub_prefix = f"{prefix}{node.name}__{attr.name}_{i}__"
                        extract_all_recursive(g, sub_prefix)

    extract_all_recursive(model.graph)

    # Filter to only the weights for the target sample rate branch
    # 16kHz uses "then_branch", 8kHz uses "else_branch"
    filtered_weights = {}
    target_prefix = f"If_0__{branch_name}__If_0_{branch_name}__Inline_0__"

    for key, value in weights.items():
        if target_prefix in key:
            # Simplify the key by removing the prefix for easier matching
            simple_key = key.replace(target_prefix, "")
            filtered_weights[simple_key] = value

    return filtered_weights


def _convert_weights_to_mlx(onnx_weights: dict, sample_rate: int = 16000) -> dict:
    """Convert ONNX weights to MLX format.

    The keys are already simplified (prefix removed by _convert_onnx_weights).
    """
    mlx_weights = {}
    hidden_size = 128

    # STFT forward basis
    stft_key = "stft.forward_basis_buffer"
    if stft_key in onnx_weights:
        # ONNX: [n_fft+2, 1, n_fft] -> MLX: [n_fft+2, n_fft, 1]
        basis = onnx_weights[stft_key].transpose(0, 2, 1)
        mlx_weights["stft.forward_basis"] = mx.array(basis)

    # Encoder conv layers
    for i in range(4):
        w_key = f"encoder.{i}.reparam_conv.weight"
        b_key = f"encoder.{i}.reparam_conv.bias"
        if w_key in onnx_weights:
            # ONNX: [out, in, kernel] -> MLX: [out, kernel, in]
            w = onnx_weights[w_key].transpose(0, 2, 1)
            mlx_weights[f"encoder.blocks.{i}.conv.weight"] = mx.array(w)
        if b_key in onnx_weights:
            mlx_weights[f"encoder.blocks.{i}.conv.bias"] = mx.array(onnx_weights[b_key])

    # LSTM weights - find by exact key match
    for k, v in onnx_weights.items():
        if "/Unsqueeze_7_output_0" in k and v.shape == (1, 512, 128):
            # W_ih: [1, 4*hidden, input] -> MLX Wx: [4*hidden, input]
            w_ih = v.squeeze(0)  # [512, 128]
            w_ih = _reorder_lstm_gates_onnx_to_mlx(w_ih, hidden_size)
            mlx_weights["decoder.lstm.Wx"] = mx.array(w_ih)
        elif "/Unsqueeze_8_output_0" in k and v.shape == (1, 512, 128):
            # W_hh: [1, 4*hidden, hidden] -> MLX Wh: [4*hidden, hidden]
            w_hh = v.squeeze(0)  # [512, 128]
            w_hh = _reorder_lstm_gates_onnx_to_mlx(w_hh, hidden_size)
            mlx_weights["decoder.lstm.Wh"] = mx.array(w_hh)
        elif "/Unsqueeze_9_output_0" in k and v.shape == (1, 1024):
            # Combined bias [1, 8*hidden] -> MLX bias [4*hidden]
            bias = v.squeeze(0)  # [1024]
            # ONNX stores Wb and Rb separately: [Wb_i,o,f,g, Rb_i,o,f,g]
            # MLX uses single bias = Wb + Rb
            b_w = bias[:512]  # Input bias
            b_r = bias[512:]  # Recurrent bias
            combined = b_w + b_r
            combined = _reorder_lstm_gates_onnx_to_mlx(combined, hidden_size)
            mlx_weights["decoder.lstm.bias"] = mx.array(combined)

    # Output conv weights
    out_w_key = "decoder.decoder.2.weight"
    out_b_key = "decoder.decoder.2.bias"
    if out_w_key in onnx_weights:
        # Output conv: ONNX [1, 128, 1] -> MLX [1, 1, 128]
        w = onnx_weights[out_w_key].transpose(0, 2, 1)
        mlx_weights["decoder.out_conv.weight"] = mx.array(w)
    if out_b_key in onnx_weights:
        mlx_weights["decoder.out_conv.bias"] = mx.array(onnx_weights[out_b_key])

    return mlx_weights


def load_vad_model(
    sample_rate: int = 16000,
    cache_dir: Optional[Path] = None,
) -> SileroVADModel:
    """Load Silero VAD model with converted weights.

    Downloads ONNX model from HuggingFace and converts to MLX format.
    Caches converted weights for faster subsequent loads.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mlx-whisper" / "vad"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    weights_file = cache_dir / f"silero_vad_{sample_rate}.npz"

    if weights_file.exists():
        weights = dict(mx.load(str(weights_file)))  # type: ignore
    else:
        onnx_weights = _convert_onnx_weights(sample_rate)
        weights = _convert_weights_to_mlx(onnx_weights, sample_rate)
        mx.savez(str(weights_file), **weights)

    model = SileroVADModel(sample_rate=sample_rate)

    # Load STFT weights
    if "stft.forward_basis" in weights:
        model.stft.forward_basis = weights["stft.forward_basis"]

    # Load encoder weights
    for i in range(4):
        w_key = f"encoder.blocks.{i}.conv.weight"
        b_key = f"encoder.blocks.{i}.conv.bias"
        if w_key in weights:
            model.encoder.blocks[i].conv.weight = weights[w_key]
        if b_key in weights:
            model.encoder.blocks[i].conv.bias = weights[b_key]

    # Load LSTM weights
    if "decoder.lstm.Wx" in weights:
        model.decoder.lstm.Wx = weights["decoder.lstm.Wx"]
    if "decoder.lstm.Wh" in weights:
        model.decoder.lstm.Wh = weights["decoder.lstm.Wh"]
    if "decoder.lstm.bias" in weights:
        model.decoder.lstm.bias = weights["decoder.lstm.bias"]

    # Load output conv weights
    if "decoder.out_conv.weight" in weights:
        model.decoder.out_conv.weight = weights["decoder.out_conv.weight"]
    if "decoder.out_conv.bias" in weights:
        model.decoder.out_conv.bias = weights["decoder.out_conv.bias"]

    mx.eval(model.parameters())
    return model
