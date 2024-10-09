import mlx_lm

model, tokenizer = mlx_lm.load("/Users/llwu/models/mlx/Meta-Llama-3.1-8B-4bit")

for s in mlx_lm.stream_generate(
    model,
    tokenizer,
    prompt=["Meta Llama 3.1 is a ", "Google Gemma 2 is a "],
    max_tokens=20,
):
    print(s[0].ljust(30) + s[1], flush=True)
