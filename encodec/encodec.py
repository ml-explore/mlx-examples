# Copyright © 2024 Apple Inc.

import functools
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

_lstm_kernel = mx.fast.metal_kernel(
    name="lstm",
    input_names=["x", "h_in", "cell", "hidden_size", "time_step", "num_time_steps"],
    output_names=["hidden_state", "cell_state"],
    header="""
    template <typename T>
    T sigmoid(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
    """,
    source="""
        uint b = thread_position_in_grid.x;
        uint d = hidden_size * 4;

        uint elem = b * d + thread_position_in_grid.y;
        uint index = elem;
        uint x_index = b * num_time_steps * d + time_step * d + index;

        auto i = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto f = sigmoid(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto g = metal::precise::tanh(h_in[index] + x[x_index]);
        index += hidden_size;
        x_index += hidden_size;
        auto o = sigmoid(h_in[index] + x[x_index]);

        cell_state[elem] = f * cell[elem] + i * g;
        hidden_state[elem] = o * metal::precise::tanh(cell_state[elem]);
    """,
)


def lstm_custom(x, h_in, cell, time_step):
    assert x.ndim == 3, "Input to LSTM must have 3 dimensions."
    out_shape = cell.shape
    return _lstm_kernel(
        inputs=[x, h_in, cell, out_shape[-1], time_step, x.shape[-2]],
        output_shapes=[out_shape, out_shape],
        output_dtypes=[h_in.dtype, h_in.dtype],
        grid=(x.shape[0], h_in.size // 4, 1),
        threadgroup=(256, 1, 1),
    )


class LSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.Wx = mx.zeros((4 * hidden_size, input_size))
        self.Wh = mx.zeros((4 * hidden_size, hidden_size))
        self.bias = mx.zeros((4 * hidden_size,)) if bias else None

    def __call__(self, x, hidden=None, cell=None):
        if self.bias is not None:
            x = mx.addmm(self.bias, x, self.Wx.T)
        else:
            x = x @ self.Wx.T

        all_hidden = []

        B = x.shape[0]
        cell = cell or mx.zeros((B, self.hidden_size), x.dtype)
        for t in range(x.shape[-2]):
            if hidden is None:
                hidden = mx.zeros((B, self.hidden_size * 4), x.dtype)
            else:
                hidden = hidden @ self.Wh.T
            hidden, cell = lstm_custom(x, hidden, cell, t)
            all_hidden.append(hidden)

        return mx.stack(all_hidden, axis=-2)


class EncodecConv1d(nn.Module):
    """Conv1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.pad_mode = config.pad_mode
        self.norm_type = config.norm_type

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, dilation=dilation
        )
        if self.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels, pytorch_compatible=True)

        self.stride = stride

        # Effective kernel size with dilations.
        self.kernel_size = (kernel_size - 1) * dilation + 1

        self.padding_total = kernel_size - stride

    def _get_extra_padding_for_conv1d(
        self,
        hidden_states: mx.array,
    ) -> mx.array:
        length = hidden_states.shape[1]
        n_frames = (length - self.kernel_size + self.padding_total) / self.stride + 1
        n_frames = int(math.ceil(n_frames)) - 1
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        return ideal_length - length

    def _pad1d(
        self,
        hidden_states: mx.array,
        paddings: Tuple[int, int],
        mode: str = "zero",
        value: float = 0.0,
    ):
        if mode != "reflect":
            return mx.pad(
                hidden_states, paddings, mode="constant", constant_values=value
            )

        length = hidden_states.shape[1]
        prefix = hidden_states[:, 1 : paddings[0] + 1][:, ::-1]
        suffix = hidden_states[:, max(length - (paddings[1] + 1), 0) : -1][:, ::-1]
        return mx.concatenate([prefix, hidden_states, suffix], axis=1)

    def __call__(self, hidden_states):
        extra_padding = self._get_extra_padding_for_conv1d(hidden_states)

        if self.causal:
            # Left padding for causal
            hidden_states = self._pad1d(
                hidden_states, (self.padding_total, extra_padding), mode=self.pad_mode
            )
        else:
            # Asymmetric padding required for odd strides
            padding_right = self.padding_total // 2
            padding_left = self.padding_total - padding_right
            hidden_states = self._pad1d(
                hidden_states,
                (padding_left, padding_right + extra_padding),
                mode=self.pad_mode,
            )

        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        return hidden_states


class EncodecConvTranspose1d(nn.Module):
    """ConvTranspose1d with asymmetric or causal padding and normalization."""

    def __init__(
        self,
        config,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ):
        super().__init__()
        self.causal = config.use_causal_conv
        self.trim_right_ratio = config.trim_right_ratio
        self.norm_type = config.norm_type
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        if config.norm_type == "time_group_norm":
            self.norm = nn.GroupNorm(1, out_channels, pytorch_compatible=True)
        self.padding_total = kernel_size - stride

    def __call__(self, hidden_states):
        hidden_states = self.conv(hidden_states)

        if self.norm_type == "time_group_norm":
            hidden_states = self.norm(hidden_states)

        if self.causal:
            padding_right = math.ceil(self.padding_total * self.trim_right_ratio)
        else:
            padding_right = self.padding_total // 2

        padding_left = self.padding_total - padding_right

        end = hidden_states.shape[1] - padding_right
        hidden_states = hidden_states[:, padding_left:end, :]
        return hidden_states


class EncodecLSTM(nn.Module):
    def __init__(self, config, dimension):
        super().__init__()
        self.lstm = [LSTM(dimension, dimension) for _ in range(config.num_lstm_layers)]

    def __call__(self, hidden_states):
        h = hidden_states
        for lstm in self.lstm:
            h = lstm(h)
        return h + hidden_states


class EncodecResnetBlock(nn.Module):
    """
    Residual block from SEANet model as used by EnCodec.
    """

    def __init__(self, config, dim: int, dilations: List[int]):
        super().__init__()
        kernel_sizes = (config.residual_kernel_size, 1)
        if len(kernel_sizes) != len(dilations):
            raise ValueError("Number of kernel sizes should match number of dilations")

        hidden = dim // config.compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [nn.ELU()]
            block += [
                EncodecConv1d(config, in_chs, out_chs, kernel_size, dilation=dilation)
            ]
        self.block = block

        if getattr(config, "use_conv_shortcut", True):
            self.shortcut = EncodecConv1d(config, dim, dim, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def __call__(self, hidden_states):
        residual = hidden_states
        for layer in self.block:
            hidden_states = layer(hidden_states)

        return self.shortcut(residual) + hidden_states


class EncodecEncoder(nn.Module):
    """SEANet encoder as used by EnCodec."""

    def __init__(self, config):
        super().__init__()
        model = [
            EncodecConv1d(
                config, config.audio_channels, config.num_filters, config.kernel_size
            )
        ]
        scaling = 1

        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config, current_scale, [config.dilation_growth_rate**j, 1]
                    )
                ]
            model += [nn.ELU()]
            model += [
                EncodecConv1d(
                    config,
                    current_scale,
                    current_scale * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            scaling *= 2

        model += [EncodecLSTM(config, scaling * config.num_filters)]
        model += [nn.ELU()]
        model += [
            EncodecConv1d(
                config,
                scaling * config.num_filters,
                config.hidden_size,
                config.last_kernel_size,
            )
        ]

        self.layers = model

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecDecoder(nn.Module):
    """SEANet decoder as used by EnCodec."""

    def __init__(self, config):
        super().__init__()
        scaling = int(2 ** len(config.upsampling_ratios))
        model = [
            EncodecConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]

        model += [EncodecLSTM(config, scaling * config.num_filters)]

        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model += [nn.ELU()]
            model += [
                EncodecConvTranspose1d(
                    config,
                    current_scale,
                    current_scale // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            ]
            for j in range(config.num_residual_layers):
                model += [
                    EncodecResnetBlock(
                        config, current_scale // 2, (config.dilation_growth_rate**j, 1)
                    )
                ]
            scaling //= 2

        model += [nn.ELU()]
        model += [
            EncodecConv1d(
                config,
                config.num_filters,
                config.audio_channels,
                config.last_kernel_size,
            )
        ]
        self.layers = model

    def __call__(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class EncodecEuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance."""

    def __init__(self, config):
        super().__init__()
        self.embed = mx.zeros((config.codebook_size, config.codebook_dim))

    def quantize(self, hidden_states):
        embed = self.embed.T
        scaled_states = hidden_states.square().sum(axis=1, keepdims=True)
        dist = -(
            scaled_states
            - 2 * hidden_states @ embed
            + embed.square().sum(axis=0, keepdims=True)
        )
        embed_ind = dist.argmax(axis=-1)
        return embed_ind

    def encode(self, hidden_states):
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))
        embed_ind = self.quantize(hidden_states)
        embed_ind = embed_ind.reshape(*shape[:-1])
        return embed_ind

    def decode(self, embed_ind):
        return self.embed[embed_ind]


class EncodecVectorQuantization(nn.Module):
    """
    Vector quantization implementation. Currently supports only euclidean distance.
    """

    def __init__(self, config):
        super().__init__()
        self.codebook = EncodecEuclideanCodebook(config)

    def encode(self, hidden_states):
        return self.codebook.encode(hidden_states)

    def decode(self, embed_ind):
        return self.codebook.decode(embed_ind)


class EncodecResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer."""

    def __init__(self, config):
        super().__init__()
        self.codebook_size = config.codebook_size

        hop_length = np.prod(config.upsampling_ratios)
        self.frame_rate = math.ceil(config.sampling_rate / hop_length)
        self.num_quantizers = int(
            1000 * config.target_bandwidths[-1] // (self.frame_rate * 10)
        )
        self.layers = [
            EncodecVectorQuantization(config) for _ in range(self.num_quantizers)
        ]

    def get_num_quantizers_for_bandwidth(
        self, bandwidth: Optional[float] = None
    ) -> int:
        """Return num_quantizers based on specified target bandwidth."""
        bw_per_q = math.log2(self.codebook_size) * self.frame_rate
        num_quantizers = self.num_quantizers
        if bandwidth is not None and bandwidth > 0.0:
            num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
        return num_quantizers

    def encode(
        self, embeddings: mx.array, bandwidth: Optional[float] = None
    ) -> mx.array:
        """
        Encode a given input array with the specified frame rate at the given
        bandwidth. The RVQ encode method sets the appropriate number of
        quantizers to use and returns indices for each quantizer.
        """
        num_quantizers = self.get_num_quantizers_for_bandwidth(bandwidth)
        residual = embeddings
        all_indices = []
        for layer in self.layers[:num_quantizers]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = mx.stack(all_indices, axis=1)
        return out_indices

    def decode(self, codes: mx.array) -> mx.array:
        """Decode the given codes to the quantized representation."""
        quantized_out = None
        for i, indices in enumerate(codes.split(codes.shape[1], axis=1)):
            layer = self.layers[i]
            quantized = layer.decode(indices.squeeze(1))
            if quantized_out is None:
                quantized_out = quantized
            else:
                quantized_out = quantized + quantized_out
        return quantized_out


class EncodecModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = EncodecEncoder(config)
        self.decoder = EncodecDecoder(config)
        self.quantizer = EncodecResidualVectorQuantizer(config)

    def _encode_frame(
        self, input_values: mx.array, bandwidth: float, padding_mask: mx.array
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Encodes the given input using the underlying VQVAE.
        """
        length = input_values.shape[1]
        duration = length / self.config.sampling_rate

        if (
            self.config.chunk_length_s is not None
            and duration > 1e-5 + self.config.chunk_length_s
        ):
            raise RuntimeError(
                f"Duration of frame ({duration}) is longer than chunk {self.config.chunk_length_s}"
            )

        scale = None
        if self.config.normalize:
            # if the padding is non zero
            input_values = input_values * padding_mask[..., None]
            mono = mx.sum(input_values, axis=2, keepdims=True) / input_values.shape[2]
            scale = mono.square().mean(axis=1, keepdims=True).sqrt() + 1e-8
            input_values = input_values / scale

        embeddings = self.encoder(input_values)
        codes = self.quantizer.encode(embeddings, bandwidth)
        return codes, scale

    def encode(
        self,
        input_values: mx.array,
        padding_mask: mx.array = None,
        bandwidth: Optional[float] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        """
        Encodes the input audio waveform into discrete codes.

        Args:
            input_values (mx.array): The input audio waveform with shape
                ``(batch_size, channels, sequence_length)``.
            padding_mask (mx.array): Padding mask used to pad the ``input_values``.
            bandwidth (float, optional): The target bandwidth. Must be one of
                ``config.target_bandwidths``. If ``None``, uses the smallest
                possible bandwidth. bandwidth is represented as a thousandth of
                what it is, e.g. 6kbps bandwidth is represented as bandwidth == 6.0

        Returns:
            A list of frames containing the discrete encoded codes for the
            input audio waveform, along with rescaling factors for each chunk
            when ``config.normalize==True``. Each frame is a tuple ``(codebook,
            scale)``, with ``codebook`` of shape ``(batch_size, num_codebooks,
            frames)``.
        """

        if bandwidth is None:
            bandwidth = self.config.target_bandwidths[0]
        if bandwidth not in self.config.target_bandwidths:
            raise ValueError(
                f"This model doesn't support the bandwidth {bandwidth}. "
                f"Select one of {self.config.target_bandwidths}."
            )

        _, input_length, channels = input_values.shape

        if channels < 1 or channels > 2:
            raise ValueError(
                f"Number of audio channels must be 1 or 2, but got {channels}"
            )

        chunk_length = self.chunk_length
        if chunk_length is None:
            chunk_length = input_length
            stride = input_length
        else:
            stride = self.chunk_stride

        if padding_mask is None:
            padding_mask = mx.ones(input_values.shape[:2], dtype=mx.bool_)
        encoded_frames = []
        scales = []

        step = chunk_length - stride
        if (input_length % stride) != step:
            raise ValueError(
                "The input length is not properly padded for batched chunked "
                "encoding. Make sure to pad the input correctly."
            )

        for offset in range(0, input_length - step, stride):
            mask = padding_mask[:, offset : offset + chunk_length].astype(mx.bool_)
            frame = input_values[:, offset : offset + chunk_length]
            encoded_frame, scale = self._encode_frame(frame, bandwidth, mask)
            encoded_frames.append(encoded_frame)
            scales.append(scale)

        encoded_frames = mx.stack(encoded_frames)

        return (encoded_frames, scales)

    @staticmethod
    def _linear_overlap_add(frames: List[mx.array], stride: int):
        if len(frames) == 0:
            raise ValueError("`frames` cannot be an empty list.")

        dtype = frames[0].dtype
        N, frame_length, C = frames[0].shape
        total_size = stride * (len(frames) - 1) + frames[-1].shape[1]

        time_vec = mx.linspace(0, 1, frame_length + 2, dtype=dtype)[1:-1]
        weight = 0.5 - (time_vec - 0.5).abs()

        weight = weight[:, None]
        sum_weight = mx.zeros((total_size, 1), dtype=dtype)
        out = mx.zeros((N, total_size, C), dtype=dtype)
        offset = 0

        for frame in frames:
            frame_length = frame.shape[1]
            out[:, offset : offset + frame_length] += weight[:frame_length] * frame
            sum_weight[offset : offset + frame_length] += weight[:frame_length]
            offset += stride

        return out / sum_weight

    def _decode_frame(
        self, codes: mx.array, scale: Optional[mx.array] = None
    ) -> mx.array:
        embeddings = self.quantizer.decode(codes)
        outputs = self.decoder(embeddings)
        if scale is not None:
            outputs = outputs * scale
        return outputs

    @property
    def channels(self):
        return self.config.audio_channels

    @property
    def sampling_rate(self):
        return self.config.sampling_rate

    @property
    def chunk_length(self):
        if self.config.chunk_length_s is None:
            return None
        else:
            return int(self.config.chunk_length_s * self.config.sampling_rate)

    @property
    def chunk_stride(self):
        if self.config.chunk_length_s is None or self.config.overlap is None:
            return None
        else:
            return max(1, int((1.0 - self.config.overlap) * self.chunk_length))

    def decode(
        self,
        audio_codes: mx.array,
        audio_scales: Union[mx.array, List[mx.array]],
        padding_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        """
        Decodes the given frames into an output audio waveform.

        Note that the output might be a bit bigger than the input. In that
        case, any extra steps at the end should be trimmed.

        Args:
            audio_codes (mx.array): Discret code embeddings of shape
                ``(batch_size, nb_chunks, chunk_length)``.
            audio_scales (mx.array): Scaling factor for each input.
            padding_mask (mx.array): Padding mask.
        """
        chunk_length = self.chunk_length
        if chunk_length is None:
            if audio_codes.shape[1] != 1:
                raise ValueError(f"Expected one frame, got {len(audio_codes)}")
            audio_values = self._decode_frame(audio_codes[:, 0], audio_scales[0])
        else:
            decoded_frames = []

            for frame, scale in zip(audio_codes, audio_scales):
                frames = self._decode_frame(frame, scale)
                decoded_frames.append(frames)

            audio_values = self._linear_overlap_add(
                decoded_frames, self.chunk_stride or 1
            )

        # truncate based on padding mask
        if padding_mask is not None and padding_mask.shape[1] < audio_values.shape[1]:
            audio_values = audio_values[:, : padding_mask.shape[1]]
        return audio_values

    @classmethod
    def from_pretrained(cls, path_or_repo: str):
        from huggingface_hub import snapshot_download

        path = Path(path_or_repo)
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=path_or_repo,
                    allow_patterns=["*.json", "*.safetensors", "*.model"],
                )
            )

        with open(path / "config.json", "r") as f:
            config = SimpleNamespace(**json.load(f))

        model = EncodecModel(config)
        model.load_weights(str(path / "model.safetensors"))
        processor = functools.partial(
            preprocess_audio,
            sampling_rate=config.sampling_rate,
            chunk_length=model.chunk_length,
            chunk_stride=model.chunk_stride,
        )
        mx.eval(model)
        return model, processor


def preprocess_audio(
    raw_audio: Union[mx.array, List[mx.array]],
    sampling_rate: int = 24000,
    chunk_length: Optional[int] = None,
    chunk_stride: Optional[int] = None,
):
    r"""
    Prepare inputs for the EnCodec model.

    Args:
        raw_audio (mx.array or List[mx.array]): The sequence or batch of
            sequences to be processed.
        sampling_rate (int): The sampling rate at which the audio waveform
            should be digitalized.
        chunk_length (int, optional): The model's chunk length.
        chunk_stride (int, optional): The model's chunk stride.
    """
    if not isinstance(raw_audio, list):
        raw_audio = [raw_audio]

    raw_audio = [x[..., None] if x.ndim == 1 else x for x in raw_audio]

    max_length = max(array.shape[0] for array in raw_audio)
    if chunk_length is not None:
        max_length += chunk_length - (max_length % chunk_stride)

    inputs = []
    masks = []
    for x in raw_audio:
        length = x.shape[0]
        mask = mx.ones((length,), dtype=mx.bool_)
        difference = max_length - length
        if difference > 0:
            mask = mx.pad(mask, (0, difference))
            x = mx.pad(x, ((0, difference), (0, 0)))
        inputs.append(x)
        masks.append(mask)
    return mx.stack(inputs), mx.stack(masks)
