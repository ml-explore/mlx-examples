# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_map


class TestModels(unittest.TestCase):

    def model_test_runner(self, model, model_type, vocab_size, num_layers):

        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs, cache = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs, cache = model(
                mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache
            )
            self.assertEqual(outputs.shape, (1, 1, vocab_size))
            self.assertEqual(outputs.dtype, t)

    def test_llama(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = llama.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi2(self):
        from mlx_lm.models import phi

        args = phi.ModelArgs()
        model = phi.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi3(self):
        from mlx_lm.models import phi3

        args = phi3.ModelArgs(
            model_type="phi3",
            hidden_size=3072,
            num_hidden_layers=32,
            intermediate_size=8192,
            num_attention_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32064,
        )
        model = phi3.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gemma(self):
        from mlx_lm.models import gemma

        args = gemma.ModelArgs(
            model_type="gemma",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            head_dim=128,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            num_key_value_heads=4,
        )
        model = gemma.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_mixtral(self):
        from mlx_lm.models import mixtral

        # Make a baby mixtral, because it will actually do the
        # eval
        args = mixtral.ModelArgs(
            model_type="mixtral",
            vocab_size=100,
            hidden_size=32,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_experts_per_tok=2,
            num_key_value_heads=2,
            num_local_experts=4,
        )
        model = mixtral.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    @unittest.skip("requires ai2-olmo")
    def test_olmo(self):
        from mlx_lm.models import olmo

        args = olmo.ModelArgs(
            model_type="olmo",
            d_model=1024,
            n_layers=4,
            mlp_hidden_size=2048,
            n_heads=2,
            vocab_size=10_000,
            embedding_size=10_000,
        )
        model = olmo.Model(args)
        self.model_test_runner(
            model,
            args.model_type,
            args.vocab_size,
            args.n_layers,
        )

    def test_qwen2_moe(self):
        from mlx_lm.models import qwen2_moe

        args = qwen2_moe.ModelArgs(
            model_type="qwen2_moe",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            num_experts_per_tok=4,
            num_experts=16,
            moe_intermediate_size=1024,
            shared_expert_intermediate_size=2048,
        )
        model = qwen2_moe.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen2(self):
        from mlx_lm.models import qwen2

        args = qwen2.ModelArgs(
            model_type="qwen2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = qwen2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_qwen(self):
        from mlx_lm.models import qwen

        args = qwen.ModelArgs(
            model_type="qwen",
        )
        model = qwen.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_plamo(self):
        from mlx_lm.models import plamo

        args = plamo.ModelArgs(
            model_type="plamo",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=8,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
        )
        model = plamo.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_stablelm(self):
        from mlx_lm.models import stablelm

        args = stablelm.ModelArgs(
            model_type="stablelm",
            vocab_size=10_000,
            hidden_size=1024,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            partial_rotary_factor=1.0,
            intermediate_size=2048,
            layer_norm_eps=1e-2,
            rope_theta=10_000,
            use_qkv_bias=False,
        )
        model = stablelm.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

        # StableLM 2
        args = stablelm.ModelArgs(
            model_type="stablelm",
            vocab_size=10000,
            hidden_size=512,
            num_attention_heads=8,
            num_hidden_layers=4,
            num_key_value_heads=2,
            partial_rotary_factor=0.25,
            intermediate_size=1024,
            layer_norm_eps=1e-5,
            rope_theta=10000,
            use_qkv_bias=True,
        )
        model = stablelm.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_starcoder2(self):
        from mlx_lm.models import starcoder2

        args = starcoder2.ModelArgs(
            model_type="starcoder2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            num_key_value_heads=4,
        )
        model = starcoder2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_cohere(self):
        from mlx_lm.models import cohere

        args = cohere.ModelArgs(
            model_type="cohere",
        )
        model = cohere.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )


if __name__ == "__main__":
    unittest.main()
