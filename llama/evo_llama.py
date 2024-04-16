# Copyright © 2023 Apple Inc.

import argparse
import math
import numpy as np
from sentencepiece import SentencePieceProcessor
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

SCALE = np.random.choice([0.025, 0.05, 0.1, 0.2])
cache_size = 0
cache_modifier = None


class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


class Llama(nn.Module):
    def __init__(
        self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.out_proj(x)

    def generate(self, x, temp=1.0):
        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.embedding(x)
        global cache_size
        global cache_modifier
        cache_size = 0
        index = 0
        print("Applying the cache modifier...")
        for i, l in enumerate(self.layers):
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            # print(f"Layer {i} ==> {len(c)} {[c[i].shape for i in range(len(c))]}")
            c = list(c)
            for j in range(len(c)):
                assert len(c[j].shape) == 4 and c[j].shape[0] == 1
                local_cache_size = c[j].shape[1] * c[j].shape[3]
                # print(f"cache{i}{j} has shape {c[j].shape} with size {local_cache_size}")
                cache_size += local_cache_size
                local_cache_modifier = cache_modifier[index : index + local_cache_size]
                assert local_cache_size + index <= len(
                    cache_modifier
                ), f"{index + local_cache_size} <= {len(cache_modifier)}"
                c[j][:, :, -1, :] += mx.array(
                    local_cache_modifier.reshape(c[j][:, :, -1, :].shape)
                )
                index += local_cache_size
            c = tuple(c)
            cache.append(c)
        assert (
            index == len(cache_modifier) or cache_modifier is None
        ), f"{len(cache_modifier) if cache_modifier is not None else None} vs {index}"
        print(f"Cache size = {cache_size}")
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.out_proj(x[:, -1])
        y = mx.random.categorical(y * (1 / temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.embedding(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))

            yield y


def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def generate(args):

    input("Press enter to start generation")
    print("------")

    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(args.prompt)])
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(x, args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= args.num_tokens:
            break

        elif (len(tokens) % args.write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], end="", flush=True)
    print()
    print("------")
    print(prompt_processing)
    print(full_gen)


def few_shot_generate(args):
    def possible_end(s):
        word = "[Instruction]"
        for i in range(len(word) - 1, 0, -1):
            if s[-i:] == word[:i]:
                return 0
        if s[-len(word) :] == word:
            return 1
        return -1

    def generate(question):
        x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(question)])
        skip = 0
        prompt_processing = None
        tokens = []
        start = tic()
        returned_value = ""
        for token in model.generate(x, args.temp):
            tokens.append(token)

            if len(tokens) == 1:
                # Actually perform the computation to measure the prompt processing time
                mx.eval(token)
                prompt_processing = toc("Prompt processing", start)

            if len(tokens) >= args.num_tokens:
                break

            mx.eval(tokens)
            token_list = [t.item() for t in tokens]
            s = tokenizer.decode(token_list)

            end = possible_end(s)
            if end == 0:
                continue
            if end == 1:
                skip = len(s)
                break

            # print(s[skip:], end="", flush=True)
            returned_value += f"{s[skip:]}"
            skip = len(s)
            if token_list[-1] == tokenizer.eos_id():
                break

        mx.eval(tokens)
        full_gen = toc("Full generation", start)
        s = tokenizer.decode([t.item() for t in tokens])
        returned_value += f"{s[skip:]}"
        print("GENERATED:", returned_value)
        print("len = ", len(returned_value))
        return returned_value
        # print(s[skip:], end="", flush=True)

    prompt = open(args.prompt).read().strip()
    import nevergrad as ng

    # optim = ng.optimizers.registry["DiscreteLenglerOnePlusOne"](86507520, 10000000)
    algorithm = "OLNDiscreteOnePlusOne"
    algorithm = "LognormalDiscreteOnePlusOne"
    algorithm = "custom"
    algorithm = "SQOPSO"
    budget = np.random.choice([10, 20, 50, 100])
    print("Budget = ", budget)
    # size = 93847552
    size = 262144
    optim = (
        ng.optimizers.registry[algorithm](size, budget)
        if algorithm != "custom"
        else ng.optimizers.Chaining(
            [
                ng.optimizers.LognormalDiscreteOnePlusOne,
                ng.optimizers.OLNDiscreteOnePlusOne,
            ],
            ["half"],
        )(size, budget)
    )
    # TODO1: questions for the training
    questions = [
        "Can you give me a list of 5 countries ?",
        "Can you give me a list of 5 famous people ?",
        "Can you give me a list of 5 events in history ?",
    ]
    # TODO2: questions for the test 
    questions2 = [
        "Can you give me a list of 5 French first names ?",
        "Can you give me a list of 5 German first names ?",
        "Can you give me a list of 5 great movies ?",
    ]
    for k in range(budget):
        global cache_modifier
        cache_modifier_candidate = optim.ask() if k < budget - 1 else optim.recommend()
        if k >= budget - 1:
             print("We recommmend")
        # question = np.random.choice(questions)   #input("Ask a question: ")
        loss = 0
        for q in questions:
            question = (
                q
                + " Please just reply by Yes or No, followed by a concrete example if your answer is Yes."
            )
            cache_modifier = SCALE * (cache_modifier_candidate.value.copy())
            diffsize = np.sum(cache_modifier != 0) / len(cache_modifier)
            answer = generate(prompt.replace("{}", question))
            cache_modifier = 0 * cache_modifier
            # TODO3: computation of the loss in training.
            # new_question = f"Do you think that << {answer} >> is a concrete answer to << {question} >> ? Please just reply 'Yes' or 'No'."
            # new_answer = generate(prompt.replace("{}", new_question))
            # print("New answer:", new_answer, ">>>>>>>")
            # if "Yes" in new_answer:
            #    loss = loss - 1
            #    print("youpi!")
            # if "No" in new_answer or "Yes" not in answer:
            #    loss = loss + 1
            #    print("bouuuuu!")
            loss = len(answer)
        optim.tell(cache_modifier_candidate, loss)
        print("we get loss ", loss)
        # optim.tell(cache_modifier_candidate, -float(input("Score between 0 and 9 ?")))
        print()
        if np.max(cache_modifier) == 0 and np.min(cache_modifier) == 0:
            print(666, "===>", loss, "(", "ZERO", ")  SCORE LLAMA+NG scale=", SCALE)
        if k < budget - 1:
            continue
        lossg = 0
        for q in questions2:
            question = (
                q
                + " Please just reply by Yes or No, followed by a concrete example if your answer is Yes."
            )
            cache_modifier = SCALE * (cache_modifier_candidate.value.copy())
            diffsize = np.sum(cache_modifier != 0) / len(cache_modifier)
            answer = generate(prompt.replace("{}", question))
            cache_modifier = 0 * cache_modifier
            # TODO4: computation of the loss in test 
            # (typically the same as in TODO3)
            loss = len(answer)
        if np.max(cache_modifier) == 0 and np.min(cache_modifier) == 0:
            print(666, "===>G", lossg, "(", "ZERO", ")  SCORE LLAMA+NG scale=", SCALE)


    print(budget, "===>", loss, "(", algorithm, ")  SCORE LLAMA+NG scale=", SCALE)
    print(f"data[{algorithm}][{budget}] += [{loss}]")
    print(f"datadiff[{algorithm}][{budget}] += [{diffsize}]")
    print(budget, "===>G", lossg, "(", algorithm, ")  SCORE LLAMA+NG scale=", SCALE)
    print(f"datag[{algorithm}][{budget}] += [{lossg}]")
    #print(f"datadiff[{algorithm}][{budget}] += [{diffsize}]")


def load_model(model_path):
    weights = mx.load(model_path)
    mlp_dims, dims = weights["layers.0.linear1.weight"].shape
    num_heads = dims // 128
    num_layers = max(int(l.split(".")[1]) for l in weights.keys() if "layers" in l) + 1
    vocab_size = weights["out_proj.weight"].shape[-1]
    model = Llama(num_layers, vocab_size, dims, mlp_dims, num_heads)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Llama inference script")
    parser.add_argument("model", help="The model file containing MLX weights")
    parser.add_argument("tokenizer", help="The sentencepiece tokenizer")
    parser.add_argument("prompt", help="The message to be processed by the model")
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Read a few shot prompt from a file (as in `sample_prompt.txt`).",
    )
    parser.add_argument(
        "--num-tokens", "-n", type=int, default=500, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.8, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    tokenizer = SentencePieceProcessor(model_file=args.tokenizer)
    print("[INFO] Loading model from disk.")
    model = load_model(args.model)
    if args.few_shot:
        few_shot_generate(args)
    else:
        generate(args)
