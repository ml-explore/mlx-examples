from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import transformers
from model import Model


class Tokenizer:
    def __init__(self, model_name: str):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            legacy=False,
            model_max_length=512,
        )
        self._decoder_start_id = 0

    @property
    def eos_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def decoder_start_id(self) -> int:
        return self._decoder_start_id

    def encode(self, s: str) -> mx.array:
        return mx.array(
            self._tokenizer(
                s,
                return_tensors="np",
                return_attention_mask=False,
            )[
                "input_ids"
            ].squeeze(0)
        )

    def decode(self, t: List[int]) -> str:
        return self._tokenizer.decode(t)


class SpeculativeDecoder:
    def __init__(
        self,
        model: Model,
        draft_model: Model,
        tokenizer: str,
        num_draft: int = 5,
        delta: float = 0.0,
    ):
        self.tokenizer = Tokenizer(tokenizer)
        self.model = model
        self.draft_model = draft_model
        self.num_draft = num_draft
        self.delta = delta

    def _generate(
        self,
        x: mx.array,
        memory: mx.array,
        draft: bool = False,
    ):
        model = self.draft_model if draft else self.model
        while True:
            logits = model.decode(x[None], memory)[0, -1]
            x = mx.argmax(logits, keepdims=True)
            lognorm = mx.logsumexp(logits.astype(mx.float32))
            logprob = logits[x] - lognorm
            yield x, logprob

    def generate(
        self,
        prompt,
        max_tokens: int = 100,
    ):
        memory = self.model.encode(self.tokenizer.encode(prompt)[None])
        x = mx.array([self.tokenizer.decoder_start_id])
        skip = 0
        outputs = []
        for (token, _), n in zip(self._generate(x, memory), range(max_tokens)):
            if token == self.tokenizer.eos_id:
                break
            outputs.append(token.item())
            if (n + 1) % 10 == 0:
                str_output = self.tokenizer.decode(outputs)
                print(str_output[skip:], end="", flush=True)
                skip = len(str_output)

        print(self.tokenizer.decode(outputs)[skip:], end="", flush=True)
        print()
        self.model.reset_cache()

    # Accept / Reject criteria (see Section 2.3 https://arxiv.org/pdf/2211.17192.pdf)
    def _get_num_accept(self, draft_tokens, draft_probs, model_logits):
        # accept_toks = mx.argmax(model_logits, axis=-1) == draft_tokens
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

        prompt = mx.array(self.tokenizer.encode(prompt), mx.uint32)[None]
        memory = self.model.encode(prompt)
        draft_memory = self.draft_model.encode(prompt)

        tokens = mx.array([self.tokenizer.decoder_start_id])

        n_steps = 0
        n_generated = 0
        n_accepted = 0
        n_drafted = 0

        outputs = []
        skip = 0
        draft_inputs = tokens
        inputs = tokens
        while True:
            # For each decoding step: generate n tokens from a draft model
            draft_tokens = []
            draft_probs = []
            for _, (t, p) in zip(
                range(n_generated, min(n_generated + self.num_draft, max_tokens)),
                self._generate(draft_inputs, draft_memory, draft=True),
            ):
                draft_tokens.append(t)
                draft_probs.append(p)
                if t.item() == self.tokenizer.eos_id:
                    break

            # Verify the draft tokens with the last verified token:
            draft_tokens = mx.concatenate(draft_tokens)
            draft_probs = mx.concatenate(draft_probs)
            verify_tokens = mx.concatenate([inputs, draft_tokens])
            logits = self.model.decode(
                verify_tokens[None, :],
                memory,
            ).squeeze(0)

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

            n_accepted += num_to_accept
            n_drafted += draft_tokens.size

            # Rewind the cache for unaccepted tokens:
            if (n := draft_tokens.size) > num_to_accept:
                self.draft_model.truncate_cache(n - new_tokens.size)
                self.model.truncate_cache(n - new_tokens.size + 1)

            n_steps += 1

            truncated = False
            for t in new_tokens.tolist():
                if t == self.tokenizer.eos_id or n_generated >= max_tokens:
                    truncated = True
                    break
                outputs.append(t)
                n_generated += 1

            str_output = self.tokenizer.decode(outputs)
            self.color = True
            if self.color and not truncated:
                model_token = len(self.tokenizer.decode(outputs[-1]))
                print(
                    "\033[34m"
                    + str_output[skip:-model_token]
                    + "\033[30m",
                    end="",
                )
                print(str_output[-model_token:], end="", flush=True)
            elif self.color and truncated:
                if truncated:
                    print(
                    "\033[34m"
                    + str_output[skip:]
                    + "\033[30m",
                    end="",
                )
            else:
                print(str_output[skip:], end="", flush=True)
            #print(str_output[skip:], end="", flush=True)
            skip = len(str_output)

            if n_generated >= max_tokens or new_tokens[-1] == self.tokenizer.eos_id:
                break
            draft_inputs = new_tokens[max(new_tokens.size - 2, 0) :]
            inputs = draft_inputs[-1:]

        print(self.tokenizer.decode(outputs)[skip:], end="", flush=True)
        print()

        self.model.reset_cache()
        self.draft_model.reset_cache()
        return {"n_accepted": n_accepted, "n_draft": n_drafted, "n_steps": n_steps}


########################################################


class PromptLookupDecoder:
    def __init__(
        self,
        model: Model,
        tokenizer: str,
        n_draft: int,
        ngram_max: int,
        ngram_min: int,
        temp: float,
        seed: int,
        color: bool,
    ):
        self.model = model
        self.tokenizer = Tokenizer(tokenizer)
        self.n_draft = n_draft
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.temp = temp
        self.seed = seed
        self.color = color

    def generate_draft(self, input_ids):
        ngram = input_ids[-self.ngram_max :]

        largest_match = 0
        draft = mx.array([], dtype=mx.uint32)

        # Sliding window search
        for i in range(1, input_ids.size - self.ngram_max):
            matches = input_ids[max(0, i - self.ngram_max) : i] == ngram[-i:]

            # reverse through the matches array
            match_length = 0
            for j in range(matches.size - 1, -1, -1):
                if matches[j]:
                    match_length += 1
                else:
                    break

            if match_length >= self.ngram_min and match_length > largest_match:
                largest_match = match_length
                start_idx = i
                end_idx = start_idx + self.n_draft
                draft = input_ids[start_idx:end_idx]

        return draft

    def prompt_lookup(
        self,
        prompt: str,
        max_tokens: int,
    ):
        def sample(logits):
            if self.temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / self.temp))

        prompt = mx.array(self.tokenizer.encode(prompt), mx.uint32)[None]
        memory = self.model.encode(prompt)

        history = prompt.squeeze(0)[
            :-1
        ]  # remove eos token from prompt lookup search space

        n_steps = 0
        n_generated = 0
        n_accepted = 0
        n_drafted = 0

        outputs = []
        skip = 0
        inputs = mx.array([self.tokenizer.decoder_start_id])
        while True:
            # For each decoding step: generate n_draft tokens by searching the prompt
            draft_tokens = self.generate_draft(history)

            # Verify draft tokens with the last verified token
            verify_tokens = mx.concatenate([inputs, draft_tokens])
            logits = self.model.decode(verify_tokens[None], memory)

            # Only keep samples that match the draft:
            # draft tokens aren't sampled - hence no accept / reject critera
            sampled = sample(logits).squeeze(0)
            equal_toks = sampled[:-1] == draft_tokens
            num_to_accept = (equal_toks.tolist() + [False]).index(False)
            new_tokens = sampled[
                : max(1, num_to_accept + 1)
            ]  # accepted draft tokens + next token from main model

            n_accepted += num_to_accept
            n_drafted += draft_tokens.size

            # Rewind the cache for unaccepted tokens:
            if (n := draft_tokens.size) > num_to_accept:
                self.model.truncate_cache(n - new_tokens.size + 1)

            n_steps += 1

            truncated = False
            for t in new_tokens.tolist():
                if t == self.tokenizer.eos_id or n_generated >= max_tokens:
                    truncated = True
                    break
                outputs.append(t)
                n_generated += 1

            str_output = self.tokenizer.decode(outputs)
            #print(str_output[skip:], end="", flush=True)


            if self.color and not truncated:
                model_token = len(self.tokenizer.decode(outputs[-1]))
                print(
                    "\033[34m"
                    + str_output[skip:-model_token]
                    + "\033[30m",
                    end="",
                )
                print(str_output[-model_token:], end="", flush=True)
            elif self.color and truncated:
                if truncated:
                    print(
                    "\033[34m"
                    + str_output[skip:]
                    + "\033[30m",
                    end="",
                )
            else:
                print(str_output[skip:], end="", flush=True)
            skip = len(str_output)

            if n_generated >= max_tokens or new_tokens[-1] == self.tokenizer.eos_id:
                break

            history = mx.concatenate([history, new_tokens])
            inputs = history[-1:]

        print(self.tokenizer.decode(outputs)[skip:], end="", flush=True)
        print()

        self.model.reset_cache()

        return {"n_accepted": n_accepted, "n_draft": n_drafted, "n_steps": n_steps}
