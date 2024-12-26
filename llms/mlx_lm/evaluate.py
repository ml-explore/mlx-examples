# Adapted from a PyTorch implementation by David Grangier

import argparse
import json
import logging
import os
from importlib.metadata import version
from pathlib import Path
from typing import Optional

import lm_eval
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from .models.cache import make_prompt_cache
from .utils import load, stream_generate

PAD = 0


def _len_longest_common_prefix(a, b):
    l = 0
    for item_a, item_b in zip(a, b):
        if item_a != item_b:
            break
        l += 1
    return l


def _rstrip_until(s, untils):
    """Limit a string <s> to the first occurrence of any substring in untils."""
    l = len(s)
    f = [s.find(u) for u in untils]
    f = [l if x < 0 else x for x in f]
    return s[: min(f)]


def _pad_inputs(
    inputs,
    maxlen,
    genlen=0,
    pad_left=False,
    pad_multiple=32,
    truncate=False,
):
    # pad the prompts to the left with at least genlen tokens.
    actual_maxlen = max(len(p) for p in inputs) + genlen
    if actual_maxlen > maxlen:
        if not truncate:
            raise ValueError("Inputs are too long.")
        else:  # drop begining
            actual_maxlen = maxlen
            inputs = [p[max(0, len(p) - maxlen) :] for p in inputs]
    if pad_multiple > 0:
        maxlen = (actual_maxlen + pad_multiple - 1) // pad_multiple
        maxlen *= pad_multiple
    assert PAD == 0
    lr = np.array((1, 0) if pad_left else (0, 1))
    return np.stack(
        [np.pad(np.array(x, np.int32), lr * (maxlen - len(x))) for x in inputs],
        axis=0,
    )


@register_model("mlxlm")
class MLXLM(LM):
    def __init__(
        self,
        path_or_hf_repo: str,
        batch_size: int = 16,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._batch_size = batch_size
        self._model, self._tokenizer = load(path_or_hf_repo)
        self._max_tokens = max_tokens or self._tokenizer.model_max_length

    def _score_fn(self, inputs, tokenize=True, step_size=32):
        if tokenize:
            inputs = self._tokenizer.encode(inputs)
        inputs = _pad_inputs(inputs, self._max_tokens, truncate=False)
        inputs = mx.array(inputs)
        inputs, targets = inputs[..., :-1], inputs[..., 1:]

        cache = make_prompt_cache(self._model)

        mask = targets != PAD

        scores, is_greedy = [], []
        for i in range(0, inputs.shape[1], step_size):
            logits = self._model(inputs[:, i : i + step_size], cache=cache)

            log_probs = nn.log_softmax(logits.astype(mx.float32))
            score = mx.take_along_axis(
                log_probs, targets[:, i : i + step_size, mx.newaxis], axis=-1
            )[..., 0]
            ig = mask[:, i : i + step_size] * (
                targets[:, i : i + step_size] == mx.argmax(logits, axis=-1)
            )

            mx.eval(score, ig)
            mx.metal.clear_cache()

            is_greedy.append(ig)
            scores.append(score)

        scores = mx.concatenate(scores, axis=1)
        is_greedy = mx.concatenate(is_greedy, axis=1)

        return scores, mask.sum(axis=-1), is_greedy

    def _loglikelihood(self, texts, score_spans=None, tokenize=True):
        # sort by length to get batches with little padding.
        sorted_indices = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        sorted_inputs = [texts[sorted_indices[i]] for i in range(len(texts))]
        sorted_spans = None
        if score_spans is not None:
            sorted_spans = [score_spans[sorted_indices[i]] for i in range(len(texts))]

        results = []
        for i in tqdm(range(0, len(sorted_inputs), self._batch_size)):
            batch = sorted_inputs[i : i + self._batch_size]
            scores, length, is_greedy = self._score_fn(batch, tokenize=tokenize)
            for j in range(len(batch)):
                if sorted_spans is None:  # full sequence score
                    mask = mx.arange(scores[j].shape[-1]) < length
                    score = (scores[j].astype(mx.float32) * mask).sum(axis=-1)
                    ig = (is_greedy[j].astype(mx.int32) * mask).sum(axis=-1)
                else:  # subsequence score
                    start, end = sorted_spans[i + j]
                    score = scores[j][start:end].astype(mx.float32).sum()
                    ig = is_greedy[j][start:end].astype(mx.int32).sum()
                    length = end - start

                results.append((score.item(), ig.item(), length))

        # reorder the outputs
        inv_sort = np.argsort(sorted_indices)
        results = [results[inv_sort[i]] for i in range(len(results))]

        return results

    def _tokenize(self, texts):
        return [tuple(self._tokenizer.encode(t)) for t in texts]

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.
        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.
        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        logging.info("Estimating loglikelihood for %d pairs." % len(requests))

        # tokenize prefix and prefix + completion for all requests.
        tokenized = self._tokenize(
            [t for r in requests for t in [r.args[0], r.args[0] + r.args[1]]]
        )

        # max length (prefix + completion) and longest common prefix per question.
        length_stats = {}
        for prefix, completed in zip(tokenized[0::2], tokenized[1::2]):
            max_completed_l, min_prefix_l = length_stats.get(prefix, (0, 1e8))
            length_stats[prefix] = (
                max(max_completed_l, len(completed)),
                min(min_prefix_l, _len_longest_common_prefix(prefix, completed)),
            )

        # truncate requests for completed sequences longer than model context.
        shortened = []
        completion_spans = []
        long_completions = 0
        for prefix, completed in zip(tokenized[0::2], tokenized[1::2]):
            max_completed_l, prefix_l = length_stats[prefix]
            # compute truncation length
            truncation = max(0, max_completed_l - self._max_tokens - 1)
            prefix_l = prefix_l - truncation
            if prefix_l <= 0:
                # completion too long, prefix is eliminated for some requests.
                long_completions += 1
                truncation = max(0, len(completed) - self._max_tokens - 1)
                prefix_l = 1
            # truncate the completed sequence
            completed = completed[truncation:]
            shortened.append(completed)
            # scores do not include initial bos, substract 1 to span bounds
            completion_spans.append((prefix_l - 1, len(completed) - 1))

        if long_completions > 0:
            logging.info(
                f"Prefix eliminated for {long_completions} requests with "
                + "completion longer than context."
            )

        # model scoring, returns num_requests x (logp, is_greedy, length).
        results = self._loglikelihood(
            shortened,
            score_spans=completion_spans,
            tokenize=False,
        )
        return [(r[0], r[1] == r[2]) for r in results]

    def loglikelihood_rolling(self, requests) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: EOT
            Max context length: 4
            Resulting input/prediction pairs:
                INPUT:  EOT   0   1   2
                PRED:     0   1   2   3
                INPUT:    3   4   5   6
                PRED:     4   5   6   7
                INPUT:    5   6   7   8
                PRED:             8   9
          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the EOT token.
        """
        logging.info(
            "Estimating loglikelihood rolling for %d sequences." % len(requests)
        )
        inputs = [req.args[0] for req in requests]
        return [t[0] for t in self._loglikelihood(inputs)]

    def generate_until(self, requests) -> list[str]:
        """Generate greedily until a stopping sequence
        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context, until).
            context: str
                Context string
            until: [str]
                The string sequences to generate until. These string sequences
                may each span across multiple tokens, or may be part of one token.
        :return: list[str]
            A list of strings continuation
            continuation: str
                The generated continuation.
        """
        logging.info("Generating continuation for %d sequences." % len(requests))
        contexts, options = zip(*[req.args for req in requests])
        # contrary to the doc the second element of the tuple contains
        # {'do_sample': False, 'until': ['\n\n'], 'temperature': 0}
        keys = list(options[0].keys())
        assert "until" in keys
        untils = [x["until"] for x in options]
        completions = []
        for context, until in tqdm(zip(contexts, untils), total=len(contexts)):
            if (
                hasattr(self._tokenizer, "apply_chat_template")
                and self._tokenizer.chat_template is not None
            ):
                messages = [{"role": "user", "content": context}]
                context = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

            max_tokens = min(
                self._max_tokens,
                self._tokenizer.model_max_length - len(self._tokenizer.encode(context)),
            )
            text = ""
            for response in stream_generate(
                self._model, self._tokenizer, prompt=context, max_tokens=max_tokens
            ):
                text += response.text
                if any(u in text for u in until):
                    text = _rstrip_until(text, until)
                    completions.append(text)
                    break
            else:
                completions.append(text)
        return completions


def main():
    parser = argparse.ArgumentParser(
        "Evaluate an MLX model using lm-evaluation-harness."
    )
    parser.add_argument("--model", help="Model to evaluate", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument(
        "--output-dir", default=".", help="Output directory for result files."
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-shots", type=int, default=0, help="Number of shots")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum nunber of tokens to generate. Defaults to the model's max context length.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Silence tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    mx.random.seed(args.seed)

    lm = MLXLM(args.model, batch_size=args.batch_size, max_tokens=args.max_tokens)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=args.tasks,
        num_fewshot=args.num_shots,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )

    model_name = args.model.replace("/", "_")
    task_names = "_".join(args.tasks)
    ver = version("lm_eval")
    filename = f"eval_{model_name}_{task_names}_{args.num_shots:02d}_v_{ver}.json"
    output_path = output_dir / filename
    output_path.write_text(json.dumps(results["results"], indent=4))
    print("Results:")
    for result in results["results"].values():
        print(json.dumps(result, indent=4))
