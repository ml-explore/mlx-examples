import mlx_lm
import random
import string

model, tokenizer = mlx_lm.load("/Users/llwu/models/mlx/Qwen2-0.5B-8bit-Instruct")

capital_letters = string.ascii_uppercase
distinct_pairs = [
    (a, b) for i, a in enumerate(capital_letters) for b in capital_letters[i + 1 :]
]

num_prompts = 16
prompt_template = "Think of a real word containing both the letters {l1} and {l2}. Then, say 3 sentences which use the word."
prompts = [
    prompt_template.format(l1=p[0], l2=p[1])
    for p in random.sample(distinct_pairs, num_prompts)
]
prompts = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    "Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works 10 hours every day for 5 days, how much money does she make?"
]
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for prompt in prompts
]

response = mlx_lm.batch_generate(
    model,
    tokenizer,
    prompts=prompts,
    max_tokens=512,
    verbose=True,
    temp=1.0,
    min_p=0.1,
    repetition_penalty=1.2,
)
