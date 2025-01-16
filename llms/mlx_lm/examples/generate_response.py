# Copyright © 2024 Apple Inc.

from mlx_lm import generate, load

# Specify the checkpoint
checkpoint = "mistralai/Mistral-7B-Instruct-v0.3"

# Load the corresponding model and tokenizer
model, tokenizer = load(path_or_hf_repo=checkpoint)

# Specify the prompt and conversation history
prompt = "Why is the sky blue?"
conversation = [{"role": "user", "content": prompt}]

# Transform the prompt into the chat template
prompt = tokenizer.apply_chat_template(
    conversation=conversation, add_generation_prompt=True
)

# Specify the maximum number of tokens
max_tokens = 1_000

# Specify if tokens and timing information will be printed
verbose = True

# Generate a response with the specified settings
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=max_tokens,
    verbose=verbose,
)
