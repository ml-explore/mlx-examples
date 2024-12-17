# Copyright © 2024 Apple Inc.
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map
from mlx_lm.models import rope_utils
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache


class TestModels(unittest.TestCase):

    def test_kv_cache(self):
        cache = KVCache()

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

    def test_rotating_kv_cache(self):
        b, h, d = 1, 2, 32
        cache = RotatingKVCache(max_size=8, step=4)

        k = mx.random.uniform(shape=(b, h, 2, d))
        v = mx.random.uniform(shape=(b, h, 2, d))

        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))
        self.assertEqual(cache.offset, 2)

        k = mx.random.uniform(shape=(b, h, 5, d))
        v = mx.random.uniform(shape=(b, h, 5, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up[..., 2:, :], k))
        self.assertTrue(mx.array_equal(v_up[..., 2:, :], v))

        k = mx.random.uniform(shape=(b, h, 4, d))
        v = mx.random.uniform(shape=(b, h, 4, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up[..., -4:, :], k))
        self.assertTrue(mx.array_equal(v_up[..., -4:, :], v))

        idx = 0
        for _ in range(10):
            k = mx.random.uniform(shape=(b, h, 1, d))
            v = mx.random.uniform(shape=(b, h, 1, d))
            k_up, v_up = cache.update_and_fetch(k, v)
            self.assertTrue(mx.array_equal(k_up[..., idx : idx + 1, :], k))
            self.assertTrue(mx.array_equal(v_up[..., idx : idx + 1, :], v))
            idx += 1
            idx %= 8

        # Try with nonzero keep
        cache = RotatingKVCache(max_size=8, step=4, keep=2)

        # Check a large update
        k = mx.random.uniform(shape=(b, h, 20, d))
        v = mx.random.uniform(shape=(b, h, 20, d))
        k_up, v_up = cache.update_and_fetch(k, v)
        self.assertTrue(mx.array_equal(k_up, k))
        self.assertTrue(mx.array_equal(v_up, v))

        # A bunch of small updates
        self.assertEqual(cache.offset, 20)
        idx = 2
        for i in range(10):
            k = mx.random.uniform(shape=(b, h, 1, d))
            v = mx.random.uniform(shape=(b, h, 1, d))
            k_up, v_up = cache.update_and_fetch(k, v)
            self.assertTrue(mx.array_equal(k_up[..., idx : idx + 1, :], k))
            self.assertTrue(mx.array_equal(v_up[..., idx : idx + 1, :], v))
            self.assertEqual(cache.offset, 21 + i)
            idx += 1
            if idx >= 8:
                idx = 2

    def test_rotating_kv_cache_chat_mode(self):
        # Test that the rotating kv cache can handle
        # alternating prompt/prefill with generation
        d = 4
        h = 2
        cache = RotatingKVCache(max_size=18, step=4)

        x = mx.random.uniform(shape=(1, h, 8, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 8)
        self.assertEqual(cache.offset, 8)

        x = mx.random.uniform(shape=(1, h, 1, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 9)
        self.assertEqual(cache.offset, 9)
        self.assertTrue(mx.allclose(x, k[..., 8:9, :]))

        x = mx.random.uniform(shape=(1, h, 2, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 11)
        self.assertEqual(cache.offset, 11)
        self.assertTrue(mx.allclose(x, k[..., 9:11, :]))

        x = mx.random.uniform(shape=(1, h, 3, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(k.shape[2], 14)
        self.assertEqual(cache.offset, 14)
        self.assertTrue(mx.allclose(x, k[..., 11:14, :]))

        x = mx.random.uniform(shape=(1, h, 6, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(cache.offset, 20)
        self.assertTrue(mx.allclose(x, k[..., -6:, :]))

        x = mx.random.uniform(shape=(1, h, 2, d))
        k, v = cache.update_and_fetch(x, x)
        self.assertEqual(cache.offset, 22)
        self.assertTrue(mx.allclose(x, k[..., -2:, :]))

    def test_rope(self):
        rope = rope_utils.initialize_rope(32, base=100, traditional=False)
        self.assertTrue(isinstance(rope, nn.RoPE))

        rope = rope_utils.initialize_rope(
            32,
            base=100,
            traditional=False,
            scaling_config={"rope_type": "linear", "factor": 10.0},
        )
        self.assertTrue(isinstance(rope, nn.RoPE))

        rope = rope_utils.initialize_rope(
            32,
            base=100,
            traditional=False,
            scaling_config={"rope_type": "llama3", "factor": 2.0},
        )
        self.assertTrue(isinstance(rope, rope_utils.Llama3RoPE))

    def model_test_runner(self, model, model_type, vocab_size, num_layers):

        self.assertEqual(len(model.layers), num_layers)
        self.assertEqual(model.model_type, model_type)

        for t in [mx.float32, mx.float16]:
            model.update(tree_map(lambda p: p.astype(t), model.parameters()))

            inputs = mx.array([[0, 1]])
            outputs = model(inputs)
            self.assertEqual(outputs.shape, (1, 2, vocab_size))
            self.assertEqual(outputs.dtype, t)

            cache = make_prompt_cache(model)
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

    def test_mamba(self):
        from mlx_lm.models import mamba

        args = mamba.ModelArgs(
            model_type="mamba",
            vocab_size=10000,
            use_bias=False,
            use_conv_bias=True,
            conv_kernel=4,
            hidden_size=768,
            num_hidden_layers=24,
            state_size=16,
            intermediate_size=1536,
            time_step_rank=48,
        )
        model = mamba.Model(args)
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

    def test_gpt_neox(self):
        from mlx_lm.models import gpt_neox

        args = gpt_neox.ModelArgs(
            model_type="gpt_neox",
            max_position_embeddings=2048,
            hidden_size=6144,
            num_attention_heads=64,
            num_hidden_layers=44,
            layer_norm_eps=1e-5,
            vocab_size=50432,
            rotary_emb_base=10_000,
            rotary_pct=0.25,
        )
        model = gpt_neox.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

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

    def test_llama3_1(self):
        from mlx_lm.models import llama

        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=1024,
            num_hidden_layers=4,
            intermediate_size=2048,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=10_000,
            max_position_embeddings=128,
            mlp_bias=False,
            num_key_value_heads=2,
            rope_scaling={
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3",
            },
        )
        model = llama.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_deepseek(self):
        from mlx_lm.models import deepseek

        args = deepseek.ModelArgs(
            model_type="deepseek",
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
        )
        model = deepseek.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_deepseek_v2(self):
        from mlx_lm.models import deepseek_v2

        args = deepseek_v2.ModelArgs(
            model_type="deepseek_v2",
            vocab_size=1024,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            kv_lora_rank=4,
            q_lora_rank=4,
            qk_rope_head_dim=32,
            v_head_dim=16,
            qk_nope_head_dim=32,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        )
        model = deepseek_v2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gemma2(self):
        from mlx_lm.models import gemma2

        args = gemma2.ModelArgs(
            model_type="gemma2",
            hidden_size=128,
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=2,
            head_dim=32,
            rms_norm_eps=1e-4,
            vocab_size=1024,
            num_key_value_heads=2,
        )
        model = gemma2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_gpt_bigcode(self):
        from mlx_lm.models import gpt_bigcode

        args = gpt_bigcode.ModelArgs(
            model_type="gpt_bigcode",
            n_embd=128,
            n_layer=128,
            n_inner=256,
            n_head=4,
            n_positions=1000,
            layer_norm_epsilon=1e-5,
            vocab_size=1024,
        )
        model = gpt_bigcode.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.n_layer)

    def test_nemotron(self):
        from mlx_lm.models import nemotron

        args = nemotron.ModelArgs(
            model_type="nemotron",
            hidden_size=128,
            hidden_act="gelu",
            num_hidden_layers=4,
            intermediate_size=256,
            num_attention_heads=4,
            norm_eps=1e-5,
            vocab_size=1024,
            num_key_value_heads=2,
        )
        model = nemotron.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phi3small(self):
        from mlx_lm.models import phi3small

        args = phi3small.ModelArgs(
            model_type="phi3small",
            hidden_size=128,
            dense_attention_every_n_layers=2,
            ff_intermediate_size=256,
            gegelu_limit=1.0,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-4,
            vocab_size=1000,
        )
        model = phi3small.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_phimoe(self):
        from mlx_lm.models import phimoe

        args = phimoe.ModelArgs(
            model_type="phimoe",
            vocab_size=320,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            rope_scaling={
                "long_factor": [1.0] * 16,
                "long_mscale": 1.243163121016122,
                "original_max_position_embeddings": 4096,
                "short_factor": [1.0] * 16,
                "short_mscale": 1.243163121016122,
                "type": "longrope",
            },
        )
        model = phimoe.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_recurrent_gemma(self):
        from mlx_lm.models import recurrent_gemma

        args = recurrent_gemma.ModelArgs(
            model_type="recurrent_gemma",
            hidden_size=128,
            attention_bias=False,
            conv1d_width=3,
            intermediate_size=256,
            logits_soft_cap=1.0,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            attention_window_size=1024,
            vocab_size=1000,
            block_types=["recurrent", "recurrent", "attention"],
        )
        model = recurrent_gemma.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_hunyuan(self):
        from mlx_lm.models import hunyuan

        args = hunyuan.ModelArgs(
            model_type="hunyuan",
            hidden_size=128,
            attention_bias=False,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            vocab_size=1000,
            moe_topk=2,
            num_experts=2,
            num_shared_expert=1,
            use_mixed_mlp_moe=True,
            use_qk_norm=True,
            rope_scaling={
                "alpha": 1000.0,
                "factor": 1.0,
                "type": "dynamic",
            },
            use_cla=True,
            cla_share_factor=2,
        )
        model = hunyuan.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_olmo2(self):
        from mlx_lm.models import olmo2

        args = olmo2.ModelArgs(
            model_type="olmo2",
            hidden_size=128,
            attention_bias=False,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=4,
            num_key_value_heads=2,
            rms_norm_eps=1e-4,
            rope_theta=1000,
            vocab_size=1000,
        )
        model = olmo2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )

    def test_exaone(self):
        from mlx_lm.models import exaone

        args = exaone.ModelArgs(
            model_type="exaone",
            hidden_size=128,
            num_layers=4,
            intermediate_size=256,
            num_attention_heads=8,
            num_key_value_heads=2,
            vocab_size=1000,
            layer_norm_epsilon=1e-4,
            rope_theta=10000,
        )
        model = exaone.Model(args)
        self.model_test_runner(model, args.model_type, args.vocab_size, args.num_layers)

    def test_cohere2(self):
        from mlx_lm.models import cohere2

        args = cohere2.ModelArgs(
            model_type="cohere2",
            hidden_size=4096,
            head_dim=128,
            num_hidden_layers=40,
            sliding_window=4096,
            sliding_window_pattern=4,
        )
        model = cohere2.Model(args)
        self.model_test_runner(
            model, args.model_type, args.vocab_size, args.num_hidden_layers
        )


if __name__ == "__main__":
    unittest.main()
