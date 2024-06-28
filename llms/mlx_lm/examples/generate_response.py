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
    conversation=conversation, tokenize=False, add_generation_prompt=True
)

# Specify the maximum number of tokens
max_tokens = 1_000

# Specify if tokens and timing information will be printed
verbose = True

# Some optional arguments for causal language model generation
generation_args = {
    "temp": 0.7,
    "repetition_penalty": 1.2,
    "repetition_context_size": 20,
    "top_p": 0.95,
}

# Generate a response with the specified settings
response = generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=max_tokens,
    verbose=verbose,
    **generation_args,
)
