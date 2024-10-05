# Copyright © 2024 Apple Inc.

"""
An example of a multi-turn chat with prompt caching.
"""

from mlx_lm import generate, load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Make the initial prompt cache for the model
prompt_cache = make_prompt_cache(model)

# User turn
prompt = "Hi my name is <Name>."
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=1024,
    temp=0.0,
    prompt_cache=prompt_cache,
)
messages.append({"role": "assistant", "content": response})

# User turn
prompt = "What's my name?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=1024,
    temp=0.0,
    prompt_cache=prompt_cache,
)
