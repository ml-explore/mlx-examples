# Learned quantization using AWQ and TesseraQ:

# References:
# AWQ:
# https://arxiv.org/abs/2306.00978
# https://github.com/mit-han-lab/llm-awq
# TesseraQ:
# https://arxiv.org/abs/2410.19103
# https://github.com/Intelligent-Computing-Lab-Yale/TesseraQ

import argparse
import glob
import shutil
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from datasets import Dataset, load_dataset
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map, tree_map_with_path
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import fetch_from_hub, get_model_path, save_config, save_weights
from tqdm import tqdm

ROUNDING_THRESHOLDS = [
    0.8,
    0.65,
    0.5,
    0.43,
    0.38,
    0.34,
    0.3,
    0.27,
    0.24,
    0.21,
    0.18,
    0.15,
    0.12,
    0.10,
    0.08,
    0.06,
    0.04,
    0.02,
    0.01,
    0.005,
]


def mse(x, y):
    return ((x - y).astype(mx.float32)) ** 2


def sigmoid(x: mx.array, gamma: float = -0.1, zeta: float = 1.1):
    return mx.clip(nn.sigmoid(x) * (zeta - gamma) + gamma, 0, 1)


def sigmoid_inverse(y: mx.array, gamma: float = -0.1, zeta: float = 1.1):
    return -mx.log((zeta - gamma) / (y - gamma) - 1)


def run_layer(layer: nn.Module, x: mx.array, batch_size: int = 32, **kwargs):
    y = []
    for i in range(0, x.shape[0], batch_size):
        y.append(layer(x[i : i + batch_size], **kwargs))
        mx.eval(y)
    y = mx.concatenate(y, axis=0)
    return y


def dist_split(x: mx.array, group: mx.distributed.Group):
    B = x.shape[0]
    N = group.size()
    assert B % N == 0
    r = group.rank()
    local_B = (B + N - 1) // N
    return x[r * local_B : (r + 1) * local_B]


def search_best_scale(
    layers: list[nn.Module],
    x: mx.array,
    quantize_func: Callable,
    block: nn.Module | None = None,
    layer_kwargs: dict | None = None,
    n_grid: int = 1,
):
    group = mx.distributed.init() if mx.distributed.is_available() else None
    layer_kwargs = layer_kwargs or {}

    block = block or layers[0]
    out = block(x, **layer_kwargs)

    x_max = x.abs().mean(axis=(0, 1))

    best_error = float("inf")
    best_scales = None

    weights = tree_flatten(block.parameters())

    for ratio in tqdm(range(n_grid)):
        ratio = ratio * 1 / n_grid
        scales = mx.maximum(x_max**ratio, 1e-4).reshape(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for layer in layers:
            layer.weight = quantize_func(layer.weight * scales) / scales

        out_q = run_layer(block, x, **layer_kwargs)
        loss = mse(out, out_q).sum()
        if group is not None:
            loss = mx.distributed.all_sum(loss, stream=mx.cpu) / group.size()
        loss /= out.size
        mx.eval(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_scales = scales

        # reload the original weights
        block.load_weights(weights)

    best_scales = best_scales.reshape(-1)
    mx.eval(best_scales)
    return best_scales


def apply_scale(prev_op, layers, scales):
    # Apply the scales to the layers
    if isinstance(prev_op, nn.Linear):
        assert len(layers) == 1
        prev_op.weight = prev_op.weight / scales[:, mx.newaxis]
        if hasattr(prev_op, "bias"):
            prev_op.bias = prev_op.bias / scales
        layers[0].weight = layers[0].weight * scales[mx.newaxis]
    elif isinstance(prev_op, (nn.LayerNorm, nn.RMSNorm)):
        prev_op.weight = prev_op.weight / scales
        if hasattr(prev_op, "bias"):
            prev_op.bias = prev_op.bias / scales
        for layer in layers:
            layer.weight = layer.weight * scales
    else:
        raise NotImplementedError(f"Could not apply scale to prev_op: {prev_op}")


def scale_block(
    block, input_feat, quantize_func: Callable, layer_kwargs: dict | None = None
):
    layers = [
        block.self_attn.q_proj,
        block.self_attn.k_proj,
        block.self_attn.v_proj,
    ]
    scales = search_best_scale(
        layers=layers,
        block=block.self_attn,
        x=input_feat["q_proj"],
        quantize_func=quantize_func,
        layer_kwargs=layer_kwargs,
    )
    apply_scale(block.input_layernorm, layers, scales)
    for name in ["q_proj", "k_proj", "v_proj"]:
        input_feat[name] = input_feat[name] / scales

    layers = [
        block.mlp.gate_proj,
        block.mlp.up_proj,
    ]
    scales = search_best_scale(
        block=block.mlp,
        layers=layers,
        x=input_feat["gate_proj"],
        quantize_func=quantize_func,
    )
    mlp_norm = getattr(
        block, "pre_feedforward_layernorm", block.post_attention_layernorm
    )
    apply_scale(mlp_norm, layers, scales)
    for name in ["gate_proj", "up_proj"]:
        input_feat[name] = input_feat[name] / scales

    layers = [block.mlp.down_proj]
    scales = search_best_scale(
        layers=layers,
        x=input_feat["down_proj"],
        quantize_func=quantize_func,
    )
    apply_scale(block.mlp.up_proj, layers, scales)
    input_feat["down_proj"] = input_feat["down_proj"] / scales


def search_best_clip(
    w: mx.array,
    x: mx.array,
    quantize_func: Callable,
    group_size: int,
    n_grid: int = 2,
    max_shrink: float = 0.5,
    subsample: int = 4,
    batch_size: int = 64,
):
    group = mx.distributed.init() if mx.distributed.is_available() else None

    x = x[:, ::subsample]
    x = x.reshape(*x.shape[:-1], -1, group_size)

    w_all = w
    w_max_all = []
    w_min_all = []

    for b in range(0, w.shape[0], batch_size):
        w = w_all[b : b + batch_size]

        group_shape = (w.shape[0], w.shape[-1] // group_size)
        best_error = mx.full(group_shape, float("inf"))
        best_w_max = mx.zeros((*group_shape, 1), dtype=x.dtype)
        best_w_min = mx.zeros((*group_shape, 1), dtype=x.dtype)

        w_shape = w.shape

        w = w.reshape(*w.shape[:-1], -1, group_size)
        out = mx.einsum("btdg,odg->btod", x, w)

        for i in range(int(max_shrink * n_grid)):
            p = 1 - i / n_grid
            w_max = p * w.max(axis=-1, keepdims=True)
            w_min = p * w.min(axis=-1, keepdims=True)
            w_m = mx.clip(w, w_min, w_max).reshape(w_shape)

            w_q = quantize_func(w_m)

            w_q = w_q.reshape(*w_q.shape[:-1], -1, group_size)
            out_q = mx.einsum("btdg,odg->btod", x, w_q)

            # Take the mean across the input batch
            loss = mse(out, out_q).sum(axis=(0, 1))
            if group is not None:
                loss = mx.distributed.all_sum(loss, stream=mx.cpu) / group.size()
            loss /= out.shape[0] * out.shape[1]
            best_indices = loss < best_error
            best_error = mx.where(best_indices, loss, best_error)
            best_w_max = mx.where(best_indices[..., mx.newaxis], w_max, best_w_max)
            best_w_min = mx.where(best_indices[..., mx.newaxis], w_min, best_w_min)
            mx.eval(best_w_max, best_w_min, best_error)

        w_max_all.append(best_w_max)
        w_min_all.append(best_w_min)

    best_w_max = mx.concatenate(w_max_all, axis=0)
    best_w_min = mx.concatenate(w_min_all, axis=0)

    w_r = w_all.reshape(*w_all.shape[:-1], -1, group_size)
    best_w = mx.clip(w_r, best_w_min, best_w_max)
    best_w = best_w.reshape(w_all.shape)

    mx.eval(best_w)
    return best_w


def clip_block(
    block: nn.Module,
    input_feat: dict[str, mx.array],
    quantize_func: Callable,
    group_size: int,
):

    def apply_clip(path, module):
        if (
            isinstance(module, nn.Linear)
            and "q_proj" not in path
            and "k_proj" not in path
        ):
            name = path.split(".")[-1]
            best_weight = search_best_clip(
                module.weight,
                input_feat[name],
                quantize_func=quantize_func,
                group_size=group_size,
            )
            module.weight = best_weight

    tree_map_with_path(apply_clip, block.leaf_modules(), is_leaf=nn.Module.is_module)


class RoundQuant(nn.Module):
    def __init__(self, module, group_size: int = 64, bits: int = 3):
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self._weight = module.weight
        if hasattr(module, "bias"):
            self._bias = module.bias

        _, self._scales, self._biases = mx.quantize(
            self._weight, group_size=group_size, bits=bits
        )
        self._scales = self._scales[..., mx.newaxis]
        self._biases = self._biases[..., mx.newaxis]

        self._weight = self._weight.reshape(self._weight.shape[0], -1, group_size)
        rounding = self._weight / self._scales
        rounding = rounding - mx.floor(rounding)
        self.rounding = sigmoid_inverse(rounding)
        self.v = mx.zeros_like(self._scales)

    def __call__(self, x: mx.array):
        q = (self._weight - self._biases) / self._scales
        q = mx.floor(q) + sigmoid(self.rounding)
        q = mx.clip(q, 0, 2**self.bits - 1)
        w = (q * self._scales * 2 * sigmoid(self.v)) + self._biases
        w = w.reshape(w.shape[0], -1)

        if hasattr(self, "_bias"):
            x = mx.addmm(self._bias, x, w.T)
        else:
            x = x @ w.T
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 3):
        assert (
            group_size == self.group_size and bits == self.bits
        ), "Quantization parameters must match"
        w = self._weight
        output_dims, input_dims = w.shape[0], w.shape[1] * w.shape[2]
        use_bias = hasattr(self, "_bias")

        q = (w - self._biases) / self._scales
        q = mx.floor(q) + sigmoid(self.rounding)
        q = mx.clip(q, 0, 2**bits - 1)
        q = q.astype(mx.uint32)

        w = q * self._scales * 2 * sigmoid(self.v) + self._biases
        w = w.reshape(w.shape[0], -1)

        q = q.reshape(q.shape[0], -1)
        bitarr = (q[..., mx.newaxis] >> mx.arange(bits, dtype=mx.uint32)) & 1
        w_q = bitarr.reshape((q.shape[0], -1, 32))
        w_q = (w_q << mx.arange(32, dtype=mx.uint32)).sum(axis=-1)

        qlayer = nn.QuantizedLinear(input_dims, output_dims, use_bias, group_size, bits)
        new_scales = self._scales * 2 * sigmoid(self.v)
        qlayer.weight = w_q
        qlayer.scales = new_scales[..., 0]
        qlayer.biases = self._biases[..., 0]
        if use_bias:
            qlayer.bias = self._bias

        return qlayer


def round_block(
    block: nn.Module,
    inputs: mx.array,
    outputs: mx.array,
    group_size: int = 64,
    bits: int = 3,
    layer_kwargs: dict | None = None,
    batch_size: int = 4,
):
    layer_kwargs = layer_kwargs or {}

    block.freeze()
    leaves = block.leaf_modules()
    rounded = tree_map(
        lambda m: RoundQuant(m, group_size, bits) if isinstance(m, nn.Linear) else m,
        leaves,
        is_leaf=nn.Module.is_module,
    )
    block.update_modules(rounded)

    def hard_round(module, threshold: float = 0):
        if not isinstance(module, RoundQuant):
            return module
        score = mx.abs(sigmoid(module.rounding) - 0.5)
        value = mx.array(np.quantile(score.astype(mx.float32), q=threshold))
        rounding = mx.where(
            sigmoid(module.rounding) > value + 0.5, float("inf"), module.rounding
        )
        module.rounding = mx.where(
            sigmoid(module.rounding) <= 0.5 - value, -float("inf"), rounding
        )
        return module

    for threshold in ROUNDING_THRESHOLDS:
        print("threshold", threshold)
        optimizer = optim.Adam(learning_rate=1e-3)

        tree_map(
            lambda m: hard_round(m, threshold),
            block.leaf_modules(),
            is_leaf=nn.Module.is_module,
        )

        def loss(block, inputs, outputs):
            outputs_q = run_layer(block, inputs, **layer_kwargs)
            return mse(outputs, outputs_q).mean()

        loss_value_and_grad = nn.value_and_grad(block, loss)

        for i in range(0, inputs.shape[0], batch_size):
            lvalue, grad = loss_value_and_grad(
                block, inputs[i : i + batch_size], outputs[i : i + batch_size]
            )
            if mx.distributed.is_available():
                grad = average_gradients(grad)
            optimizer.update(block, grad)
            mx.eval(block.parameters(), optimizer.state, lvalue)
        print(lvalue)

    tree_map(hard_round, block.leaf_modules(), is_leaf=nn.Module.is_module)


def awq_quantize(
    model,
    inputs: mx.array,
    group_size: int = 64,
    bits: int = 3,
    embed_group_size: int = 32,
    embed_bits: int = 4,
):
    group = mx.distributed.init() if mx.distributed.is_available() else None

    def quantize_func(w):
        wq = mx.quantize(w, bits=bits, group_size=group_size)
        return mx.dequantize(*wq, bits=bits, group_size=group_size)

    mask = create_attention_mask(inputs)

    model.model.embed_tokens = model.model.embed_tokens.to_quantized(
        group_size=embed_group_size, bits=embed_bits
    )
    inputs = model.model.embed_tokens(inputs)

    input_feat = {}

    def capture(path, module):
        if not isinstance(module, nn.Linear):
            return module

        class Catcher(nn.Module):
            def __call__(self, x: mx.array):
                name = path.split(".")[-1]
                input_feat[name] = x
                return module(x)

        return Catcher()

    for i, layer in enumerate(model.model.layers):
        import time

        s = time.time()
        print(f"Starting block {i}")

        # capture the inputs to each layer
        orig_leaves = layer.leaf_modules()
        capture_leaves = tree_map_with_path(
            capture, orig_leaves, is_leaf=nn.Module.is_module
        )
        layer.update_modules(capture_leaves)
        outputs = run_layer(layer, inputs, mask=mask)
        layer.update_modules(orig_leaves)
        del capture_leaves

        nn.quantize(layer, group_size=group_size, bits=bits)
        outputs_q = run_layer(layer, inputs, mask=mask)
        loss = mse(outputs, outputs_q).sum()
        if group is not None:
            loss = mx.distributed.all_sum(loss, stream=mx.cpu) / group.size()
        loss /= outputs.size
        print("Before Loss", loss, flush=True)
        layer.update_modules(orig_leaves)
        del orig_leaves

        print("Scaling block", flush=True)
        scale_block(
            block=layer,
            input_feat=input_feat,
            quantize_func=quantize_func,
            layer_kwargs={"mask": mask},
        )

        print("Clipping block", flush=True)
        clip_block(
            block=layer,
            input_feat=input_feat,
            quantize_func=quantize_func,
            group_size=group_size,
        )

        print("Rounding block")
        round_block(
            block=layer,
            inputs=inputs,
            outputs=outputs,
            group_size=group_size,
            bits=bits,
            layer_kwargs={"mask": mask},
        )

        nn.quantize(layer, group_size=group_size, bits=bits)
        outputs_q = run_layer(layer, inputs, mask=mask)
        loss = mse(outputs, outputs_q).sum()
        if group is not None:
            loss = mx.distributed.all_sum(loss, stream=mx.cpu) / group.size()
        loss /= outputs.size
        print("After Loss", loss, flush=True)

        input_feat = {}
        inputs = outputs

        mx.eval(layer)
        mx.metal.clear_cache()

        e = time.time()
        print("Loop time: ", e - s)

    if hasattr(model, "lm_head"):
        model.lm_head = model.lm_head.to_quantized(
            group_size=embed_group_size, bits=embed_bits
        )


def load_wikitext(
    tokenizer, num_samples: int = 32, sequence_length: int = 2048, split: str = "train"
) -> mx.array:
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = "\n\n".join(dataset["text"])
    tokens = tokenizer.encode(texts, return_tensors="mlx")[0]

    # Select random chunks
    starts = mx.random.randint(
        0, len(tokens) - sequence_length - 1, shape=(num_samples, 1)
    )
    data = tokens[starts + mx.arange(sequence_length)]
    if tokenizer.bos_token_id:
        data = mx.concatenate(
            [mx.full((*data.shape[:2], 1), tokenizer.bos_token_id), data], axis=-1
        )
    return data


def save_model(
    model: nn.Module,
    tokenizer: TokenizerWrapper,
    config,
    model_path: Path,
    mlx_path: str,
):
    weights = dict(tree_flatten(model.parameters()))

    mlx_path = Path(mlx_path)
    save_weights(mlx_path, weights, donate_weights=True)

    py_files = glob.glob(str(model_path / "*.py"))
    for file in py_files:
        shutil.copy(file, mlx_path)

    tokenizer.save_pretrained(mlx_path)

    config["quantization"] = {"group_size": 64, "bits": 4}

    def update_config(path, module):
        if hasattr(module, "bits"):
            config["quantization"][path] = {
                "group_size": module.group_size,
                "bits": module.bits,
            }
        else:
            config["quantization"][path] = False

    tree_map_with_path(update_config, model.leaf_modules(), is_leaf=nn.Module.is_module)

    save_config(config, config_path=mlx_path / "config.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", default="mlx-community/Qwen2.5-7B-Instruct-bf16"
    )
    parser.add_argument("--mlx-path", default="mlx_model")
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    group = mx.distributed.init() if mx.distributed.is_available() else None

    num_samples = args.num_samples
    if group is not None and num_samples % group.size() > 0:
        num_samples += group.size() - num_samples % group.size()

    mx.random.seed(args.seed)

    model_path = get_model_path(args.model, revision=None)
    model, config, tokenizer = fetch_from_hub(model_path, lazy=True)

    calibration_data = load_wikitext(tokenizer, args.num_samples, args.sequence_length)

    if group is not None:
        calibration_data = dist_split(calibration_data, group)

    awq_quantize(model, calibration_data, bits=args.bits, group_size=args.group_size)

    save_model(model, tokenizer, config, model_path, args.mlx_path)


if __name__ == "__main__":
    main()
