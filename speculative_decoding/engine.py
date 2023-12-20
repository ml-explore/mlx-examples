import transformers
from dataclasses import dataclass, field
from model import Llama
import mlx.core as mx
import mlx.nn as nn
import time
import numpy as np
from typing import Optional

from prompts import create_urial_prompt

def clone(x: mx.array):
    return mx.array(np.array(x))

@dataclass
class LlamaEngine:
    model: str # path to HuggingFace repo
    draft_model: Optional[str] = None # path to draft model
    tokenizer: transformers.AutoTokenizer = field(init=False)

    def __post_init__(self):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model)
        self.model = Llama.from_hugging_face(self.model)
        if self.draft_model is not None:
            self.draft_model = Llama.from_hugging_face(self.draft_model)
    
    def tokenize(self, messages):
        if self.tokenizer.chat_template is not None:
            tokenized = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True
            )
        else:
            # use urial zero-shot template
            tokenized = self.tokenizer.encode(create_urial_prompt(messages[0]["content"]))
        
        return tokenized

    def generate(
        self, 
        messages, 
        num_tokens: int = 100, 
        temp: float = 0.8,
        draft_model: bool = False # if true gen with draft model
    ):  
        tokenized = self.tokenize(messages)
        x = mx.array([tokenized])
        tokens = []
        start = time.time()
        for token in self.model.generate(x, temp):
            if token.item() == 2:
                break
            tokens.append(token)
            if len(tokens) >= num_tokens:
                break
        mx.eval(tokens)
        run_time = time.time() - start
        tokens = [t.item() for t in tokens]
        s = self.tokenizer.decode(tokens)
        # print("=== COMPLETION ===")
        # print(s)
        print(f"=== GENERATED {len(tokens)} TOKENS in {run_time} SECONDS ===")
        return s

    # generate only supports batch size 1, so should this
    def speculative_decode(
        self, 
        messages, 
        num_tokens: int = 100, 
        temp: float = 0.8,
        n: int = 5
    ):
        batch_size = 1
        tokenized = self.tokenize(messages)
        tokens = mx.array([tokenized])
        prompt_len = tokens.shape[1]
        start = time.time()
        
        # prefill the main model
        # sample first token & write draft from there (avoids rewinding main model which i was doing before)
        logit = self.model(tokens, read_cache=False, write_cache=True, next_token_only=True)
        first_token = mx.random.categorical(logit * (1 / temp)).reshape(batch_size, 1)
        tokens = mx.concatenate([tokens, first_token], axis=1)

        decoding_steps = 1
        n_new_tokens = 1
        accepted_draft_tokens = 0
        draft_logit = self.draft_model(tokens, read_cache=False, write_cache=True, next_token_only=True)
        # print("Before doing any speculative decoding, draft model cache is: ", self.draft_model.kv_cache[0][0].shape[2])
        # print("And prompt length is: ", prompt_len)
        while True:
            # for each decoding step: generate n tokens from a draft model
            draft_tokens = mx.random.categorical(draft_logit * (1 / temp)).reshape(batch_size, 1)
            draft_tokens_left = n - 1
            for token in self.draft_model.generate( # generate automatically updates the cache, it has to
                draft_tokens,
                temp=temp,
                read_cache=True
            ):
                draft_tokens = mx.concatenate([draft_tokens, token.reshape(batch_size, 1)], axis=1)
                draft_tokens_left -= 1
                if draft_tokens_left == 0:
                    break
            
            # have to verify the first draft token using the last verified token
            verify_tokens = mx.concatenate([tokens[:, -1:], draft_tokens], axis=1)
            # print("Tokens so far: ", self.tokenizer.decode(np.array(tokens[0, prompt_len:]).tolist()))
            # print("Predicted draft tokens: [", self.tokenizer.decode(np.array(draft_tokens[0, :]).tolist()), "]")
            logits = self.model(verify_tokens, read_cache=True, write_cache=True)
            # check the last n + 1 tokens
            logits = logits[:, -(n + 1):, :]
            sampled = mx.random.categorical(logits * (1 / temp), axis=-1)
            # print("Sampled tokens: [", self.tokenizer.decode(np.array(sampled[0, :]).tolist()), "]")

            # only keep samples that match the draft
            num_to_accept = 0
            for i in range(n):
                if mx.all(sampled[:, i] == draft_tokens[:, i]):
                    num_to_accept += 1
                else:
                    break
            # print("Accepting", num_to_accept)
            accepted_draft_tokens += num_to_accept
            n_new_tokens += (1 + num_to_accept)
            
            new_tokens = sampled[:, :num_to_accept + 1]
            tokens = mx.concatenate([tokens, new_tokens], axis=1)

            # truncate draft cache to keep only accepted tokens
            # what tokens have been INPUT into the draft model? let's say n = 5, start with |p| tokens
            # |p| -> t0; |p + t0| -> t1; |p + t0 + t1| -> t2; |p + t0 + t1 + t2| -> t3; |p + t0 + t1 + t2 + t3| -> t4;
            # return -> t0 - t4, cache has |p + t0 + t1 + t2 + t3|
            # main model accepts whatever is correct, then generates t'
            # if 0 accepted: cache should have |p + t'|
            # if 1 accepted: |p + t0 + t'|
            # if 2 accepted: |p + t0 + t1 + t'|
            # ...
            # if 5 accepted: |p + t0 + t1 + t2 + t3 + t4 + t'|
            # we're always going to have to show the draft model the 1 token where it went off
            # the rails and we rejected it and took the real model, cause that won't be in its cache
            # print("After speculative decoding, before truncation, draft cache has: ",  self.draft_model.kv_cache[0][0].shape[2])
            if num_to_accept < n:
                self.draft_model.truncate_kv_cache(n - 1 - num_to_accept)
                # print("Truncated draft cache by", n - 1 - num_to_accept, "now it has", self.draft_model.kv_cache[0][0].shape[2])
            elif num_to_accept == n:
                # forward the model on the last draft token to catch it up
                # maybe this is bad?
                self.draft_model(draft_tokens[:, -1:], read_cache=True, write_cache=True, next_token_only=True)

            # now how to truncate the full model's cache?
            # it has |p + t0 + t1 + t2 + t3 + t4|
            # if 0 accepted: truncate back to p
            # if 1 accepted: truncate to p + t0
            self.model.truncate_kv_cache(n - num_to_accept) # how many to truncate? i think 1 off from draft model? idk
            # NOTE: main model doesn't know that it predicted t' (the 1 non-draft token)
            # i think this is ok because it's the last accepted token and will be passed back in at verification time

            # NOTE: model is now (or could be!) 1 token ahead of draft model cause if it accepts the full
            # draft it's now predicted 1 token past the draft token's last token. must account for this.

            
            # print("Length of accepted tokens: ", tokens.shape[1])
            # print("Length of draft model cache: ", self.draft_model.kv_cache[0][0].shape[2])
            # print("Length of main model cache: ", self.model.kv_cache[0][0].shape[2])
            decoding_steps += 1
            
            if n_new_tokens >= num_tokens or mx.any(new_tokens == 2):
                break

            # get the next draft token based on t', preparing to do it all again!
            # print("Getting the token that comes after: ", self.tokenizer.decode(np.array(tokens[0, -1:]).tolist()))
            draft_logit = self.draft_model(tokens[:, -1:], read_cache=True, write_cache=True, next_token_only=True)

        mx.eval(tokens)
        end = time.time()
        
        seq = np.array(tokens[0, :]).tolist()
        s = self.tokenizer.decode(seq[prompt_len:])
        # print(f"=== COMPLETION {0 + 1} ===")
        # print(s)

        print("=== GENERATED", n_new_tokens, "TOKENS IN", round(end - start, 2), "SECONDS ===")
        print("=== ACCEPTED", accepted_draft_tokens, "DRAFT TOKENS ===")
        print("=== DECODING STEPS", decoding_steps, "===")
        return s

        

            