# Copyright © 2024 Apple Inc.

import copy
import os
import tempfile
import unittest

import mlx.core as mx
from mlx_lm.models.cache import (
    KVCache,
    MambaCache,
    RotatingKVCache,
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.utils import generate_step, load

HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"


class TestPromptCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_load(self):
        cache = [KVCache() for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Test with metadata
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")
        metadata = {"a": "b", "c": "d"}
        save_prompt_cache(cache_file, cache, metadata)
        _, loaded_metadata = load_prompt_cache(cache_file, return_metadata=True)
        self.assertEqual(metadata, loaded_metadata)

    def test_save_load_rotating_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        # Test with rotating cache
        cache = [RotatingKVCache(max_size=8, keep=2) for _ in range(4)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        self.assertTrue(len(cache), len(loaded_cache))
        for c, lc in zip(cache, loaded_cache):
            self.assertEqual(c.offset, lc.offset)
            self.assertEqual(c.keep, lc.keep)
            self.assertEqual(c.max_size, lc.max_size)
            self.assertEqual(c.step, lc.step)
            self.assertTrue(mx.array_equal(c.state[0], lc.state[0]))
            self.assertTrue(mx.array_equal(c.state[1], lc.state[1]))

        # Do a couple single token updates to get a rotation
        for _ in range(2):
            for c in cache:
                x = mx.random.uniform(shape=(1, 8, 1, 4))
                c.update_and_fetch(x, x)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)

        for c, lc in zip(cache, loaded_cache):
            x = mx.random.uniform(shape=(1, 8, 1, 4))
            k, v = c.update_and_fetch(x, x)
            lk, lv = lc.update_and_fetch(x, x)
            self.assertEqual(c.offset, lc.offset)
            self.assertTrue(mx.array_equal(k, lk))
            self.assertTrue(mx.array_equal(v, lv))

    def test_save_load_mixed_cache(self):
        cache_file = os.path.join(self.test_dir, "prompt_cache.safetensors")

        cache = [MambaCache(), KVCache(), RotatingKVCache(8), MambaCache()]
        for c in cache:
            if isinstance(c, MambaCache):
                c[0] = mx.random.uniform(shape=(4, 4, 4))
                c[1] = mx.random.uniform(shape=(4, 4, 4))
            else:
                x = mx.random.uniform(shape=(4, 4, 7, 4))
                y = mx.random.uniform(shape=(4, 4, 7, 4))
                c.update_and_fetch(x, y)

        save_prompt_cache(cache_file, cache)
        loaded_cache = load_prompt_cache(cache_file)
        for c, lc in zip(cache, loaded_cache):
            if isinstance(c, MambaCache):
                self.assertTrue(mx.array_equal(c[0], lc[0]))
                self.assertTrue(mx.array_equal(c[1], lc[1]))
            else:
                x = mx.random.uniform(shape=(4, 4, 1, 4))
                y = mx.random.uniform(shape=(4, 4, 1, 4))
                k, v = c.update_and_fetch(x, y)
                lk, lv = lc.update_and_fetch(x, y)
                self.assertEqual(c.offset, lc.offset)
                self.assertTrue(mx.array_equal(k, lk))
                self.assertTrue(mx.array_equal(v, lv))

    def test_cache_with_generate(self):
        model, tokenizer = load(HF_MODEL_PATH)
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]
        results = zip(range(4), generate_step(prompt, model))
        toks, all_logits = zip(*(r[1] for r in results))

        prompt_cache = make_prompt_cache(model)
        i = 0
        for _, (tok, logits) in zip(
            range(2), generate_step(prompt, model, prompt_cache=prompt_cache)
        ):
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))
            i += 1

        for _, (tok, logits) in zip(
            range(1),
            generate_step(mx.array([toks[i]]), model, prompt_cache=prompt_cache),
        ):
            i += 1
            self.assertEqual(tok, toks[i])
            self.assertTrue(mx.allclose(logits, all_logits[i]))

    def test_trim_cache(self):
        cache = [KVCache() for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        # Trim
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 7)

        # Trim more tokens than remain
        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 3)

        # Can't trim mamba cache
        cache = [MambaCache() for _ in range(2)]
        for c in cache:
            c.state = mx.zeros((5, 5))
        num_trimmed = trim_prompt_cache(cache, 7)
        self.assertEqual(num_trimmed, 0)

        # All cache's have to be trimmable
        cache = [MambaCache(), KVCache()]
        cache[0].state = mx.zeros((5, 5))
        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[1].update_and_fetch(x, x)
        num_trimmed = trim_prompt_cache(cache, 1)
        self.assertEqual(num_trimmed, 0)

        cache = [RotatingKVCache(max_size=6) for _ in range(2)]
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 5, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 4)

        # Can't trim fixed-size KV cache after processing
        # more than max_kv_size tokens
        for c in cache:
            x = mx.random.uniform(shape=(1, 8, 10, 4))
            c.update_and_fetch(x, x)

        num_trimmed = trim_prompt_cache(cache, 4)
        self.assertEqual(num_trimmed, 0)

    def test_trim_cache_with_generate(self):
        model, tokenizer = load(HF_MODEL_PATH)
        prompt = tokenizer.encode("this is a prompt", return_tensors="mlx")[0]

        prompt_cache = make_prompt_cache(model)

        # Generate one token so we process the full prompt
        last_tok, _ = next(generate_step(prompt, model, prompt_cache=prompt_cache))
        last_tok = mx.array([last_tok])

        # Generate two more tokens
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        toks, all_logits = zip(*(r[1] for r in results))

        # To get back to the cache just after processing the prompt,
        # trim by 3 tokens
        trim_prompt_cache(prompt_cache, 3)

        # Generate the same thing again
        results = zip(
            range(2), generate_step(last_tok, model, prompt_cache=prompt_cache)
        )
        second_toks, second_all_logits = zip(*(r[1] for r in results))
        self.assertEqual(toks, second_toks)
        self.assertTrue(
            all(mx.allclose(l, l2) for l, l2 in zip(all_logits, second_all_logits))
        )

    def test_cache_copying(self):
        cache = [KVCache()]

        x = mx.random.uniform(shape=(1, 8, 10, 4))
        cache[0].update_and_fetch(x, x)

        y = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(y, y)

        old_cache = copy.deepcopy(cache)

        trim_prompt_cache(cache, 1)

        self.assertTrue(old_cache[0].offset, 11)
        self.assertTrue(cache[0].offset, 10)

        z = mx.random.uniform(shape=(1, 8, 1, 4))
        cache[0].update_and_fetch(z, z)

        self.assertTrue(mx.allclose(old_cache[0].keys[..., 10:11, :], y))
        self.assertTrue(mx.allclose(cache[0].keys[..., 10:11, :], z))


if __name__ == "__main__":
    unittest.main()
