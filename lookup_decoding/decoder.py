import mlx.core as mx
from mlx.utils import tree_map
import argparse
import time
from mistral import load_model

class PromptLookupDecoder:
    def __init__(self, model: str) -> None:
        model, tokenizer = load_model(model)
        self.model = model 
        self.tokenizer = tokenizer
    
    def _generate(
        self,
        x: mx.array,
        temp: float = 0.0,
    ):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        logits, cache = self.model(x[None])
        y = sample(logits[:, -1, :])
        yield y

        while True:
            logits, cache = self.model(x[None, :], cache, next_token_only=True)
            x = sample(logits)
            yield x

    # Generate without prompt lookup decoding (for testing)
    def generate(
        self,
        prompt,
        max_tokens: int = 100,
        temp: float = 0.0,
    ):
        print("[INFO] Starting generation...")
        print(prompt, end="", flush=True)
        x = mx.array(self.tokenizer.encode(prompt), mx.uint32)

        start = time.time()
        for token, n in zip(self._generate(x, temp), range(max_tokens)):
            token = token.item()
            if token == self.tokenizer.eos_id:
                break
            print(self.tokenizer.decode([token]), end="", flush=True)
        run_time = time.time() - start
        print()
        print(f"=== GENERATED {n + 1} TOKENS IN {run_time} SECONDS ===")
    
    """
        Considerations:
        - If a match is found but we can't draft n_draft tokens, do we draft as
        many as we can or check for a match with a smaller ngram size?
        - How do we choose if there are multiple matches?
        
        This implementation:
        - Ignores a match if we can't draft n_draft tokens. This avoids the risk
        of only drafting a few tokens. 
        - We exit upon the first match. This avoids the need to rank matches.
    """
    def prompt_lookup(self, prompt: str, max_tokens: int, n_draft: int, 
                 ngram_max: int, ngram_min: int, temp: float, seed: int,
                 color: bool):
        
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))
        
        mx.random.seed(seed)

        print("[INFO] Starting generation...")
        start = time.time()
        print(prompt, end="", flush=True)
        prompt = mx.array(self.tokenizer.encode(prompt), mx.uint32)
        tokens = prompt

        # prefill model
        logit, cache = self.model(prompt[None])
        token = sample(logit[:, -1, :]) 
        tokens = mx.concatenate([tokens, token])
        prompt_time = time.time() - start
        print(self.tokenizer.decode(token.tolist()), end="", flush=True)

        n_drafted = 0
        n_accepted = 0
        n_generated = 1
        n_decoding_steps = 0

        while True:
            # For each decoding step: generate n_draft tokens by searching the prompt
            def generate_draft(input_ids):
                input_length = input_ids.size

                for ngram_size in range(ngram_max, ngram_min, -1): 
                    ngram = input_ids[-ngram_size:]

                    for i in range(input_length - ngram_size):
                        if mx.all(input_ids[i:i+ngram_size] == ngram):
                            start_idx = i + ngram_size
                            end_idx = start_idx + n_draft
                            if start_idx < input_length - ngram_size:
                                return input_ids[start_idx:end_idx]

                return mx.array([], dtype=mx.uint32)
            
            draft_tokens = generate_draft(tokens)
            n_drafted += draft_tokens.size

            # Verify draft tokens with the last verified token
            verify_tokens = mx.concatenate([tokens[-1:], draft_tokens])
            logits, cache = self.model(verify_tokens[None], cache=cache)
            sampled = sample(logits).squeeze(0)

            # Only keep samples that match the draft. 
            equal_toks = sampled[:-1] == draft_tokens
            num_to_accept = (equal_toks.tolist() + [False]).index(False)
            new_tokens = sampled[: max(1, num_to_accept + 1)]

            n_accepted += num_to_accept

            # Rewind the cache for unaccepted tokens:
            if (num_to_truncate := draft_tokens.size - num_to_accept) > 0:
                if num_to_truncate < cache[0][0].shape[2]:
                    cache = tree_map(
                        lambda x: x[:, :, :-num_to_truncate, :], cache
                    )
                else:
                    cache = [None] * len(self.model.layers)
            
            n_decoding_steps += 1

            # Check stop decodig criteria:
            for t in new_tokens.tolist()[:-1]:
                if t == self.tokenizer.eos_id:
                    break
                if (color):
                    print("\033[34m" + self.tokenizer.decode([t]) + "\033[30m", end="", flush=True)
                else:
                    print(self.tokenizer.decode([t]), end="", flush=True)

            print(self.tokenizer.decode(new_tokens[-1:].tolist()), end="", flush=True)

            n_generated += new_tokens.size
            if n_generated >= max_tokens or new_tokens[-1] == self.tokenizer.eos_id:
                break

            tokens = mx.concatenate([tokens, new_tokens])

        end = time.time()
        print()
        print("=== PROMPT EVAL IN", round(prompt_time, 2), "SECONDS ===")
        print("=== GENERATED", n_generated, "TOKENS IN", round(end - start, 2), "SECONDS ===")
        print("=== ACCEPTED", n_accepted, "DRAFT TOKENS ===")
        print("=== ACCEPT", round(n_accepted/n_generated * 100, 2), "% ===")
        print("=== DECODING STEPS", n_decoding_steps, "===")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mistral inference script")
    parser.add_argument(
        "--model-path",
        type=str,
        default="mlx_model",
        help="The path to the model weights and tokenizer",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model",
        default="This is a test. This is a test. This is a",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--n-draft",
        type=int,
        default=10,
        help="Number of draft tokens to generate upon prompt lookup match",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=3,
        help="Maximum ngrams to match against input during prompt lookup",
    )
    parser.add_argument(
        "--ngram-min",
        type=int,
        default=1,
        help="Minimum ngrams to match against input during prompt lookup",
    )
    parser.add_argument(   
        "--temp",
        help="The sampling temperature.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The PRNG seed"
    )
    parser.add_argument(
        "--color",
        type=bool,
        default=False,
        help="Color the accepted draft tokens"
    )

    args = parser.parse_args()

    mx.random.seed(args.seed)
    print("[INFO] Loading model from disk.")

    engine = PromptLookupDecoder(args.model_path)

    engine.prompt_lookup(
        args.prompt, 
        args.max_tokens, 
        args.n_draft, 
        args.ngram_max, 
        args.ngram_min, 
        args.temp, 
        args.seed,
        args.color
        )
    

