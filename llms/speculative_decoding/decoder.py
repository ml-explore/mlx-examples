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
        ntoks = 0
        n_accepted = 0
        n_draft = 0

        outputs = []
        skip = 0
        draft_inputs = tokens
        inputs = tokens
        while True:
            # For each decoding step: generate n tokens from a draft model
            draft_tokens = []
            draft_probs = []
            for _, (t, p) in zip(
                range(ntoks, min(ntoks + self.num_draft, max_tokens)),
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
            n_draft += draft_tokens.size

            # Rewind the cache for unaccepted tokens:
            if (n := draft_tokens.size) > num_to_accept:
                self.draft_model.truncate_cache(n - new_tokens.size)
                self.model.truncate_cache(n - new_tokens.size + 1)

            n_steps += 1

            for t in new_tokens.tolist():
                if t == self.tokenizer.eos_id or ntoks >= max_tokens:
                    break
                outputs.append(t)
                ntoks += 1

            str_output = self.tokenizer.decode(outputs)
            print(str_output[skip:], end="", flush=True)
            skip = len(str_output)

            if ntoks >= max_tokens or new_tokens[-1] == self.tokenizer.eos_id:
                break
            draft_inputs = new_tokens[max(new_tokens.size - 2, 0) :]
            inputs = draft_inputs[-1:]

        print(self.tokenizer.decode(outputs)[skip:], end="", flush=True)
        print()

        self.model.reset_cache()
        self.draft_model.reset_cache()
        return {"n_accepted": n_accepted, "n_draft": n_draft, "n_steps": n_steps}
