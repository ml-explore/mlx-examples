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
    def __init__(
        self,
        model: str,
        draft_model: str = None,
        num_draft: int = 5,
        delta: float = 0.0,
    ):
        self.tokenizer = Tokenizer(model)
        self.model = Llama.from_hugging_face(model)
        if draft_model is not None:
            self.draft_model = Llama.from_hugging_face(draft_model)
        self.num_draft = num_draft
        self.delta = delta

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
        draft: bool = False,
    ):
        model = self.draft_model if draft else self.model
        while True:
            logits = model(x[None, :], next_tokens=1).squeeze((0, 1))
            x = mx.argmax(logits, keepdims=True)
            lognorm = mx.logsumexp(logits.astype(mx.float32))
            logprob = logits[x] - lognorm
            yield x, logprob

    def generate(
        self,
        prompt,
        max_tokens: int = 100,
        draft: bool = False,
    ):
        x = self.tokenize(prompt)
        start = time.time()
        for (token, _), n in zip(self._generate(x, draft=draft), range(max_tokens)):
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

    def _get_num_accept(self, draft_tokens, draft_probs, model_logits):
        # equal_toks = sampled[:-1] == draft_tokens
        model_probs = mx.take_along_axis(
            model_logits,
            draft_tokens[:, None],
            axis=-1,
        ).squeeze(-1)
        model_probs -= mx.logsumexp(model_logits.astype(mx.float32), axis=-1)
        unis = mx.random.uniform(shape=(draft_tokens.size,))
        log_unis = mx.log(mx.maximum(unis - self.delta, 0.0))
        accept_toks = log_unis <= ((model_probs - draft_probs))
        num_to_accept = (accept_toks.tolist() + [False]).index(False)
        return num_to_accept

    def speculative_decode(
        self,
        prompt,
        max_tokens: int = 100,
    ):
        def sample(logits):
            return mx.argmax(logits, axis=-1)

        tokens = mx.array(self.tokenize(prompt), mx.uint32)
        start = time.time()

        decoding_steps = 0
        ntoks = 0
        accepted_draft_tokens = 0
        total_draft_tokens = 0

        draft_inputs = tokens
        inputs = tokens
        while True:
            # For each decoding step: generate n tokens from a draft model
            draft_tokens = []
            draft_probs = []
            for _, (t, p) in zip(
                range(ntoks, min(ntoks + self.num_draft, max_tokens)),
                self._generate(draft_inputs, draft=True),
            ):
                draft_tokens.append(t)
                draft_probs.append(p)
                if t.item() == self.tokenizer.eos_id:
                    break

            # Verify the draft tokens with the last verified token:
            draft_tokens = mx.concatenate(draft_tokens)
            draft_probs = mx.concatenate(draft_probs)
            verify_tokens = mx.concatenate([inputs, draft_tokens])
            logits = self.model(
                verify_tokens[None, :], next_tokens=draft_tokens.size + 1
            ).squeeze(0)
            # sampled = sample(logits).squeeze(0)

            # Only keep samples that match the draft:
            num_to_accept = self._get_num_accept(
                draft_tokens,
                draft_probs,
                logits[:-1],
            )
            new_tokens = draft_tokens[:num_to_accept]
            # Get the next token from the main model as well
            new_tokens = mx.concatenate(
                [new_tokens, mx.argmax(logits[num_to_accept], keepdims=True)]
            )

            accepted_draft_tokens += num_to_accept
            total_draft_tokens += draft_tokens.size

            # Rewind the cache for unaccepted tokens:
            if (n := draft_tokens.size) > num_to_accept:
                self.draft_model.truncate_cache(n - new_tokens.size)
                self.model.truncate_cache(n - new_tokens.size + 1)

            decoding_steps += 1

            for t in new_tokens.tolist():
                if t == self.tokenizer.eos_id or ntoks >= max_tokens:
                    break
                print(self.tokenizer.decode(t, with_sep=ntoks > 0), end="", flush=True)
                ntoks += 1
            if ntoks >= max_tokens or new_tokens[-1] == self.tokenizer.eos_id:
                break
            draft_inputs = new_tokens[max(new_tokens.size - 2, 0) :]
            inputs = draft_inputs[-1:]

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
        print(
            f"=== ACCEPTED {accepted_draft_tokens} of {total_draft_tokens} DRAFT TOKENS ==="
        )
        print("=== DECODING STEPS", decoding_steps, "===")
