# Copyright © 2024 Apple Inc.

"""
An example of a multi-turn chat with prompt caching.
"""

from mlx_lm import generate, load
from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Make the initial prompt cache for the model
prompt_cache = make_prompt_cache(model)

# User turn
prompt = "Hi my name is <Name>."
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)

# User turn
prompt = "What's my name?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Assistant response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    prompt_cache=prompt_cache,
)

# Save the prompt cache to disk to reuse it at a later time
save_prompt_cache("mistral_prompt.safetensors", prompt_cache)

# Load the prompt cache from disk
prompt_cache = load_prompt_cache("mistral_prompt.safetensors")
