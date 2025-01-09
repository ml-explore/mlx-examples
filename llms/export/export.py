import time
from pathlib import Path

import fire
import mlx.core as mx
from mlx_lm import load


class ExportableCache:

    def __init__(self, keys=None, values=None, offset=0):
        self.offset = offset
        self.keys = keys
        self.values = values

    def update_and_fetch(self, keys, values):
        if self.keys is not None:
            self.keys = mx.slice_update(self.keys, keys, self.offset, axes=(2,))
            self.values = mx.slice_update(self.values, values, self.offset, axes=(2,))
        else:
            self.keys = keys
            self.values = values
        return self.keys, self.values

    @property
    def state(self):
        return self.keys, self.values


def expand(cache, mask=None, cache_step_size=256):
    cache_size = cache[0].shape[-2]
    new_size = cache_step_size * ((cache_size + cache_step_size) // cache_step_size)

    def expand_kv(x):
        B, n_heads, _, head_dim = x.shape
        new_x = mx.zeros((B, n_heads, new_size, head_dim), x.dtype)
        new_x[..., : x.shape[2], :] = x
        return new_x

    cache = [expand_kv(c) for c in cache]
    if mask is None:
        mask = mx.full(new_size, False)
        mask[:cache_size] = True
    else:
        mask = mx.concatenate([mask, mx.full(cache_step_size, False)])
    return cache, mask


def causal_mask(N):
    idx = mx.arange(N)
    return idx[:, None] >= idx


def step(model, y, *state):
    mask = state[-1]
    if len(state) > 1:
        cache, offset = state[:-2], state[-2]
        cache = [
            ExportableCache(keys, values, offset)
            for keys, values in zip(cache[::2], cache[1::2])
        ]
    else:
        cache = [ExportableCache() for i in range(len(model.model.layers))]
    logits = model(y, cache=cache, mask=mask)
    cache = [y for x in cache for y in x.state]
    return logits, *cache


def generate_step(prompt, model, max_tokens):
    mx.eval(model)

    compiled_step = mx.compile(lambda *args: step(model, *args), shapeless=True)

    def _step(*args):
        logits, *cache = compiled_step(*args)
        return mx.argmax(logits[:, -1], axis=-1), *cache

    y, *cache = _step(prompt, causal_mask(prompt.size))
    mx.async_eval(y)
    offset = mx.array(prompt.size, mx.uint32)
    cache, mask = expand(cache)
    n = 0
    while True:
        if n < max_tokens - 1:
            if mask.size <= (prompt.size + n):
                cache, mask = expand(cache, mask)
            mask[prompt.size + n] = True
            next_y, *cache = _step(y[None], *cache, offset, mask)
            mx.async_eval(next_y)
            offset += 1
        n += 1
        yield y.item()
        if n == max_tokens:
            break
        y = next_y


def export(
    model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    path="llama3.1-instruct-4bit",
):
    model, tokenizer = load(model)

    mx.eval(model)

    tokenizer.save_pretrained(path)

    _step = lambda *args: step(model, *args)

    # Make example inputs
    y_prompt = mx.array([[0, 0]], mx.uint32)
    y_gen = mx.array([[0]], mx.uint32)
    offset = mx.array([0], mx.uint32)

    mask = causal_mask(y_prompt.size)
    _, *cache = _step(y_prompt, mask)

    model_path = str(Path(path) / "model.mlxfn")
    with mx.exporter(model_path, _step, shapeless=True) as exporter:
        exporter(y_prompt, mask)
        cache, mask = expand(cache)
        exporter(y_gen, *cache, offset, mask)


def generate(
    prompt,
    model="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    max_tokens=128,
):
    print("[INFO] Loading model from disk.")
    model, tokenizer = load(model)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="mlx",
    )

    print("[INFO] Starting generation...")
    tic = time.time()
    tokens = []

    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    for n, token in enumerate(generate_step(prompt, model, max_tokens)):
        if n == 0:
            prompt_tps = prompt.size / (time.time() - tic)
            tic = time.time()

        if token in tokenizer.eos_token_ids:
            break
        detokenizer.add_token(token)
        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()
    print(detokenizer.last_segment, flush=True)
    gen_tps = (n + 1) / (time.time() - tic)
    peak_memory = mx.metal.get_peak_memory() / 1e9
    print("=" * 10)
    print(f"Prompt: {prompt_tps:.3f} tokens-per-sec")
    print(f"Generation: {gen_tps:.3f} tokens-per-sec")
    print(f"Peak RAM: {peak_memory:.3f} GB")


if __name__ == "__main__":
    fire.Fire(
        {
            "generate": generate,
            "export": export,
        }
    )
