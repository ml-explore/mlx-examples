import math
import sys
import unittest
from io import StringIO

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import tuner
from mlx_lm.tuner.dora import DoRALinear


class TestDora(unittest.TestCase):
    def setUp(self):
        self.capturedOutput = StringIO()
        sys.stdout = self.capturedOutput

    def tearDown(self):
        sys.stdout = sys.__stdout__

    def test_llama(self):
        from mlx_lm.models import llama

        hidden_size = 1024
        intermediate_size = 2048
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=hidden_size,
            num_hidden_layers=4,
            intermediate_size=intermediate_size,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )

        dora_layers = 4

        def check_config(params):
            n_keys = 2
            if "keys" in params:
                n_keys = len(params["keys"])
            model = llama.Model(args)
            model.freeze()
            tuner.utils.linear_to_lora_layers(model, dora_layers, params, use_dora=True)
            trainable_params = sum(
                v.size for _, v in tree_flatten(model.trainable_parameters())
            )
            self.assertEqual(
                trainable_params,
                dora_layers
                * (params["rank"] * hidden_size * 2 * n_keys + n_keys * hidden_size),
            )

        params = {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0}
        check_config(params)

        params["rank"] = 1
        check_config(params)

        params["keys"] = ["self_attn.k_proj"]
        check_config(params)

    def mx_assert_equal(self, a, b):
        self.assertTrue(mx.array_equal(a, b))

    def mx_assert_not_equal(self, a, b):
        self.assertFalse(mx.array_equal(a, b))

    def test_dora_m_parameter(self):
        dora_lin = DoRALinear(input_dims=100, output_dims=100)
        self.mx_assert_equal(dora_lin.m, mx.linalg.norm(dora_lin.linear.weight, axis=1))

        # Recomputes m when changing Linear
        inital_m = dora_lin.m
        dora_lin.set_linear(nn.Linear(10, 10))
        self.mx_assert_not_equal(inital_m, dora_lin.m)
        self.mx_assert_equal(dora_lin.m, mx.linalg.norm(dora_lin.linear.weight, axis=1))

        # Works with quantized weights
        quantized_linear = nn.QuantizedLinear(512, 512)
        dora_lin.set_linear(quantized_linear)
        dequantized_weight = mx.dequantize(
            quantized_linear.weight,
            quantized_linear.scales,
            quantized_linear.biases,
            quantized_linear.group_size,
            quantized_linear.bits,
        )
        self.mx_assert_equal(dora_lin.m, mx.linalg.norm(dequantized_weight, axis=1))

    def test_dora_from_linear(self):
        in_dims = 1024
        out_dims = 512
        r = 4

        linear = nn.Linear(in_dims, out_dims)
        dora_lin = DoRALinear.from_linear(linear, r)
        self.mx_assert_equal(dora_lin.m, mx.linalg.norm(linear.weight, axis=1))
        self.assertEqual(dora_lin.lora_a.shape, (in_dims, r))
        self.assertEqual(dora_lin.lora_b.shape, (r, out_dims))
        self.assertEqual(dora_lin.m.shape, (out_dims,))

        quantized_linear = nn.QuantizedLinear(in_dims, out_dims)
        dequantized_weight = mx.dequantize(
            quantized_linear.weight,
            quantized_linear.scales,
            quantized_linear.biases,
            quantized_linear.group_size,
            quantized_linear.bits,
        )
        dora_quant_lin = DoRALinear.from_linear(quantized_linear, r)
        self.mx_assert_equal(
            dora_quant_lin.m, mx.linalg.norm(dequantized_weight, axis=1)
        )
        self.assertEqual(dora_quant_lin.lora_a.shape, (in_dims, r))
        self.assertEqual(dora_quant_lin.lora_b.shape, (r, out_dims))
        self.assertEqual(dora_quant_lin.m.shape, (out_dims,))

    def test_dora_to_linear(self):
        in_dims = 1024
        out_dims = 512
        r = 4

        linear = nn.Linear(in_dims, out_dims, bias=True)
        dora_lin = DoRALinear.from_linear(linear, r)
        to_linear = dora_lin.to_linear()
        self.mx_assert_equal(linear.weight, to_linear.weight)
        self.mx_assert_equal(linear.bias, to_linear.bias)

        def dequantize_weight(quantized_linear):
            return mx.dequantize(
                quantized_linear.weight,
                quantized_linear.scales,
                quantized_linear.biases,
                quantized_linear.group_size,
                quantized_linear.bits,
            )

        quantized_linear = nn.QuantizedLinear(in_dims, out_dims, bias=True)
        dora_quantized_linear = DoRALinear.from_linear(quantized_linear, r)
        # Dequantize
        to_linear_from_quantized = dora_quantized_linear.to_linear(de_quantize=True)
        self.mx_assert_equal(quantized_linear.bias, to_linear_from_quantized.bias)
        self.mx_assert_equal(
            dequantize_weight(quantized_linear), to_linear_from_quantized.weight
        )

    def test_dora_backprop(self):
        in_dims = 1024
        out_dims = 512
        r = 4

        linear = nn.QuantizedLinear(in_dims, out_dims, bias=True)
        dora_lin = DoRALinear.from_linear(linear, r)
        dora_lin.train()

        input = mx.random.uniform(shape=(in_dims,))
        target = mx.random.uniform(shape=(out_dims,))

        optimizer = optim.Adam(learning_rate=2e-5)

        def loss_fn(inputs, targets):
            outputs = dora_lin(inputs)
            loss = (outputs - targets).square().mean()
            return loss

        loss_value_and_grad = nn.value_and_grad(dora_lin, loss_fn)
        initial_loss = None
        for i in range(20):
            loss, grad = loss_value_and_grad(input, target)
            self.assertFalse(math.isnan(loss.item()))
            optimizer.update(dora_lin, grad)

            if i == 0:
                initial_loss = loss

        self.assertGreater(initial_loss, loss)


if __name__ == "__main__":
    unittest.main()
