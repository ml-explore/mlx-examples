import mlx_lm

# model, tokenizer = mlx_lm.load("mlx-community/SmolLM-1.7B-Instruct-fp16")
model, tokenizer = mlx_lm.load("/Users/llwu/models/mlx/Qwen2-0.5B-8bit-Instruct")
draft_model, draft_tokenizer = mlx_lm.load("mlx-community/SmolLM-135M-Instruct-4bit")

# https://github.com/hemingkx/Spec-Bench/blob/main/data/spec_bench/question.jsonl
prompt = "Develop a Python program that reads all the text files under a directory and returns top-5 words with the most number of occurrences."

prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)

mlx_lm.generate(
    model,
    tokenizer,
    prompt=prompt,
    verbose=True,
    max_tokens=500,
    temp=1.0,
    min_p=0.1,
    repetition_penalty=1.2,
    # draft_model=draft_model,
)
