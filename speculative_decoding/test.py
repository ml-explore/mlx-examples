import time
from engine import LlamaEngine

# This will use the chat template from the primary model
engine = LlamaEngine(
    model="meta-llama/Llama-2-7b-hf", 
    draft_model="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T"
)

messages = [
    {"role": "user", "content": "Finish the monologue: To be, or not to be..."}
]

# Do 1 regular generation to get warmed up (the first one is slow)
engine.generate(messages, num_tokens=1, temp=0.1)

# Time regular generation
start = time.time()
engine.generate(messages, num_tokens=125, temp=0.1)
print(f"Regular generation took {time.time() - start} seconds")

# Time speculative decoding
start = time.time()
engine.speculative_decode(messages, num_tokens=125, temp=0.1, n=5)
print(f"Speculative decoding took {time.time() - start} seconds")