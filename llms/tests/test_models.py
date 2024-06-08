# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
from mlx.utils import tree_map
from mlx_lm.models.base import KVCache


class TestModels(unittest.TestCase):

    def test_kv_cache(self):
        cache = KVCache(32, 4)

        k = mx.ones((1, 4, 1, 32), mx.float16)
        v = mx.ones((1, 4, 1, 32), mx.float16)

        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))
        self.assertEqual(cache.offset, 1)

        k = mx.ones((1, 4, cache.step, 32), mx.float16)
        v = mx.ones((1, 4, cache.step, 32), mx.float16)
        k_up, v_up = cache.update_and_fetch(k, v)

        expected = mx.ones((1, 4, cache.step + 1, 32), mx.float16)
        self.assertTrue(mx.array_equal(k_up, expected))
        self.assertTrue(mx.array_equal(v_up, expected))
        self.assertEqual(cache.offset, cache.step + 1)

    def model_test_runner(self, model, model_type, vocab_size, num_layers):

        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            kv_heads = (
                [model.n_kv_heads] * len(model.layers)
                if isinstance(model.n_kv_heads, int)
                else model.n_kv_heads
            )
            cache = [KVCache(model.head_dim, n) for n in kv_heads]

            outputs = model(inputs, cache)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            outputs = model(mx.argmax(outputs[0, -1:, :], keepdims=True), cache=cache)
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

    def test_phixtral(self):
        from mlx_lm.models import phixtral

        args = phixtral.ModelArgs(
            "phixtral", num_vocab=1000, num_layers=4, model_dim=1024
        )
        model = phixtral.Model(args)
        self.model_test_runner(model, args.model_type, args.num_vocab, args.num_layers)

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

    def test_dbrx(self):
        from mlx_lm.models import dbrx

        args = dbrx.ModelArgs(
            model_type="dbrx",
            d_model=1024,
            ffn_config={"ffn_hidden_size": 2048, "moe_num_experts": 4, "moe_top_k": 2},
            attn_config={"kv_n_heads": 2, "clip_qkv": True, "rope_theta": 10000},
            n_layers=4,
            n_heads=4,
            vocab_size=10_000,
        )
        model = dbrx.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.n_layers)

    def test_minicpm(self):
        from mlx_lm.models import minicpm

        args = minicpm.ModelArgs(
            model_type="minicpm",
            hidden_size=1024,
            dim_model_base=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-4,
            vocab_size=10000,
            num_key_value_heads=2,
            scale_depth=1.0,
            scale_emb=1.0,
        )
        model = minicpm.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gpt2(self):
        from mlx_lm.models import gpt2

        args = gpt2.ModelArgs(
            model_type="gpt2",
            n_ctx=1024,
            n_embd=768,
            n_head=12,
            n_layer=12,
            n_positions=1024,
            layer_norm_epsilon=1e-5,
            vocab_size=50256,
        )
        model = gpt2.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.n_layer)

    def test_openelm(self):
        from mlx_lm.models import openelm

        args = openelm.ModelArgs(
            model_type="openelm",
            ffn_dim_divisor=256,
            ffn_multipliers=[
                0.5,
                0.73,
                0.97,
                1.2,
                1.43,
                1.67,
                1.9,
                2.13,
                2.37,
                2.6,
                2.83,
                3.07,
                3.3,
                3.53,
                3.77,
                4.0,
            ],
            head_dim=64,
            model_dim=1280,
            normalize_qk_projections=True,
            num_kv_heads=[3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5],
            num_query_heads=[
                12,
                12,
                12,
                12,
                12,
                16,
                16,
                16,
                16,
                16,
                16,
                16,
                20,
                20,
                20,
                20,
            ],
            num_transformer_layers=16,
            vocab_size=32000,
        )

        model = openelm.Model(args)
        self.model_test_runner(
            model,
            args.model_type,
            args.vocab_size,
            len(args.ffn_multipliers),
        )

    def test_internlm2(self):
        from mlx_lm.models import internlm2

        args = internlm2.ModelArgs(
            model_type="internlm2",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10000,
        )
        model = internlm2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )


if __name__ == "__main__":
    unittest.main()
