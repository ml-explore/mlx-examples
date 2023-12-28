import time
from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import transformers
from model import Llama
from prompts import create_urial_prompt


class Tokenizer:
    def __init__(self, model_name: str):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    @property
    def eos_id(self) -> int:
        return self._tokenizer.eos_token_id

    def encode(self, s: str) -> mx.array:
        return mx.array(
            self._tokenizer(s, return_tensors="np", return_attention_mask=False,)[
                "input_ids"
            ].squeeze(0)
        )

    def decode(self, t: List[int], with_sep: bool = True) -> str:
        tokens = self._tokenizer.convert_ids_to_tokens(t)
        return "".join(t.replace("▁", " " if with_sep else "") for t in tokens)


class SpeculativeDecoder:
    def __init__(self, model: str, draft_model: str = None):
        self.tokenizer = Tokenizer(model)
        self.model = Llama.from_hugging_face(model)
        if draft_model is not None:
            self.draft_model = Llama.from_hugging_face(draft_model)

    def tokenize(self, prompt):
        # if self.tokenizer.chat_template is not None:
        #    tokenized = self.tokenizer.apply_chat_template(
        #        prompt, tokenize=True, add_generation_prompt=True
        #    )
        # else:
        # use urial zero-shot template
        tokenized = self.tokenizer.encode(create_urial_prompt(prompt["content"]))
        return tokenized

    def _generate(
        self,
        x: mx.array,
        temp: float = 0.0,
        draft: bool = False,
    ):
        model = self.draft_model if draft else self.model

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        while True:
            logit = model(x[None, :], next_token_only=True)
            x = sample(logit)
            yield x

    def generate(
        self,
        prompt,
        max_tokens: int = 100,
        temp: float = 0.0,
        draft: bool = False,
    ):
        x = self.tokenize(prompt)
        start = time.time()
        for token, n in zip(self._generate(x, temp, draft=draft), range(max_tokens)):
            token = token.item()
            if token == self.tokenizer.eos_id:
                break
            print(self.tokenizer.decode(token, with_sep=n > 0), end="", flush=True)
        run_time = time.time() - start
        print()
        print(f"=== GENERATED {n + 1} TOKENS in {run_time} SECONDS ===")
        if draft:
            self.draft_model.reset_cache()
        else:
            self.model.reset_cache()

    def speculative_decode(
        self, prompt, max_tokens: int = 100, temp: float = 0.0, n_draft: int = 5
    ):
        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        tokens = mx.array(self.tokenize(prompt), mx.uint32)
        start = time.time()

        decoding_steps = 0
        ntoks = 0
        accepted_draft_tokens = 0

        while True:
            # For each decoding step: generate n tokens from a draft model
            draft_tokens = []
            for _, t in zip(
                range(ntoks, min(ntoks + n_draft, max_tokens)),
                self._generate(tokens, temp=temp, draft=True),
            ):
                draft_tokens.append(t)
                if t.item() == self.tokenizer.eos_id:
                    break

            # Verify the draft tokens with the last verified token
            draft_tokens = mx.concatenate(draft_tokens)
            verify_tokens = mx.concatenate([tokens, draft_tokens])
            logits = self.model(verify_tokens[None, :-1])
            sampled = sample(logits[:, -draft_tokens.size :]).squeeze(0)

            # Only keep samples that match the draft:
            equal_toks = sampled == draft_tokens
            num_to_accept = (equal_toks.tolist() + [False]).index(False)
            new_tokens = sampled[: max(1, num_to_accept)]

            accepted_draft_tokens += num_to_accept

            # Rewind the cache for unaccepted tokens:
            if (n := draft_tokens.size) > num_to_accept:
                self.draft_model.truncate_cache(n - new_tokens.size)
                self.model.truncate_cache(n - new_tokens.size)

            decoding_steps += 1

            # Check stop decodig criteria:
            for t in new_tokens.tolist():
                if t == self.tokenizer.eos_id:
                    break
                print(self.tokenizer.decode(t, with_sep=ntoks > 0), end="", flush=True)
            ntoks += new_tokens.size
            if ntoks >= max_tokens or new_tokens[-1] == self.tokenizer.eos_id:
                break
            tokens = new_tokens[-1:]

        end = time.time()
        self.model.reset_cache()
        self.draft_model.reset_cache()
        print()
        print(
            "=== GENERATED",
            ntoks,
            "TOKENS IN",
            round(end - start, 2),
            "SECONDS ===",
        )
        print("=== ACCEPTED", accepted_draft_tokens, "DRAFT TOKENS ===")
        print("=== DECODING STEPS", decoding_steps, "===")
