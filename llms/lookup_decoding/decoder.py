import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_map
import argparse
import time
import json
from typing import List, Optional, Tuple
from mistral import Mistral, Tokenizer, ModelArgs
from pathlib import Path

class MistralEngine:

    def __init__(self, model: str) -> None:
        model, tokenizer = self.load_model(model)
        self.model = model 
        self.tokenizer = tokenizer
    
    def load_model(self, folder: str):
        model_path = Path(folder)
        tokenizer = Tokenizer(str(model_path / "tokenizer.model"))
        with open(model_path / "config.json", "r") as f:
            config = json.loads(f.read())
            config.pop("sliding_window", None)
            config.pop("model_type", None)
            quantization = config.pop("quantization", None)
            model_args = ModelArgs(**config)
        weights = mx.load(str(model_path / "weights.npz"))
        weights = tree_unflatten(list(weights.items()))
        model = Mistral(model_args)
        if quantization is not None:
            nn.QuantizedLinear.quantize_module(model, **quantization)
        model.update(weights)
        mx.eval(model.parameters())
        return model, tokenizer
    
    # note to self: do something about 'self'
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
    def prompt_lookup(self, input_ids, ngram_max, ngram_min, n_draft):
        input_length = input_ids.size

        for ngram_size in range(ngram_max, ngram_min, -1): 
            ngram = input_ids[0, -ngram_size:]

            for i in range(input_length - ngram_size):
                if mx.all(input_ids[0, i:i+ngram_size] == ngram):
                    start_idx = i + ngram_size
                    end_idx = start_idx + n_draft
                    if start_idx < input_length - ngram_size:
                        return input_ids[0, start_idx:end_idx]

        return mx.array([])
    
    
    def generate(self, prompt: str, max_tokens: int, n_draft: int, ngram_max: int, ngram_min: int, temp: float, seed: int):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))
        
        mx.random.seed(seed)

        print("[INFO] Starting generation...")
        tic = time.time()
        print(prompt, end="", flush=True)
        prompt = mx.array(self.tokenizer.encode(prompt))
        tokens = prompt # will store all tokens generated (for prompt lookup)

        # prefill model
        logit, cache = self.model(prompt[None])
        token = sample(logit[:, -1, :]) 
        tokens = mx.concatenate([tokens, token])

        n_drafted = 0
        n_accepted = 0
        n_generated = 1
        n_past = prompt.size

        while True:
            # For each decoding step: generate n_draft tokens by searching the prompt
            draft_tokens = self.prompt_lookup(tokens, ngram_max, ngram_min, n_draft)
            n_drafted += draft_tokens.size

            
            verify_tokens = mx.concatenate([tokens[-1], draft_tokens])
            logits, cache = self.model(verify_tokens[None, :-1], cache=cache)


            logits = logits[:, :-1, :]
            sampled = sample(logits)

            num_to_accept = 0
            for i in range(n_draft):
                if mx.all(sampled[:, i] == draft_tokens[:, i]):
                    num_to_accept += 1
                else:
                    break
            
            n_past += num_to_accept
            n_accepted += num_to_accept
            n_generated += (1 + num_to_accept)

            accepted_tokens = sampled[:, :num_to_accept + 1]
            tokens = mx.concatenate([tokens, accepted_tokens], axis=1)

            mx.eval(accepted_tokens)
            s = self.tokenizer.decode([t.item() for t in accepted_tokens])
            print(s, end="", flush=True)

            # truncate kv cache to keep only accepted tokens
            # self.model.truncate_kv_cache(n - num_to_accept)
            cache_length = cache[0][0].shape[2]
            num_to_truncate = min(num_to_truncate, cache_length)
            if num_to_truncate == 0:
                pass
            else:
                cache = tree_map(lambda x: x[:, :, :-num_to_truncate, :], cache)                
            
            if n_accepted >= max_tokens or mx.any(accepted_tokens == self.tokenizer.eos_token_id):
                break
    
        mx.eval(tokens)
        s = self.tokenizer.decode([t.item() for t in tokens])
        print(s, flush=True)
        print("------")
        generation_tps = ntoks / (time.time() - tic)
        print(
            f"Tokens per second: prompt {prompt_tps:.3f}, "
            f"generation {generation_tps:.3f}"
        )

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

    args = parser.parse_args()

    mx.random.seed(args.seed)
    print("[INFO] Loading model from disk.")

    engine = MistralEngine(args.model_path)
    engine.generate(
        args.prompt, 
        args.max_tokens, 
        args.n_draft, 
        args.ngram_max, 
        args.ngram_min, 
        args.temp, 
        args.seed
        )
    

