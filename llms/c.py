import mlx_lm

model, tokenizer = mlx_lm.load("/Users/llwu/models/mlx/Meta-Llama-3.1-8B-4bit")

for s in mlx_lm.stream_generate(
    model,
    tokenizer,
    prompt="Meta Llama 3.1 is a ",
    max_tokens=100,
):
    print(s, end="", flush=True)
