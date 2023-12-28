import time

from decoder import SpeculativeDecoder

# This will use the chat template from the primary model
engine = SpeculativeDecoder(
    # model="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
    model="meta-llama/Llama-2-7b-hf",
    draft_model="TinyLlama/TinyLlama-1.1B-intermediate-step-1195k-token-2.5T",
)

messages = {"role": "user", "content": "Finish the monologue: To be, or not to be..."}

# Do 1 regular generation to get warmed up (the first one is slow)
engine.generate(messages, max_tokens=1)
engine.generate(messages, max_tokens=1, draft=True)

# Time regular generation
engine.generate(messages, max_tokens=125)

# Time speculative decoding
engine.speculative_decode(messages, max_tokens=125, n_draft=10)
