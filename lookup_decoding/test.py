from decoder import PromptLookupDecoder

prompt = "[INST] Repeat the following phrase 10 times: 'The quick brown fox jumps over the lazy dog.'. Don't say antyhing else. [/INST] "

engine = PromptLookupDecoder("mlx_model")

engine.generate(prompt, 250)

engine.prompt_lookup(prompt, 250, 10, 3, 1, 0.0, 0, True)
