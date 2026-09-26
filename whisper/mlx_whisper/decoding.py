# Copyright © 2023 Apple Inc.

import zlib
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def detect_language(
    model: "Whisper", mel: mx.array, tokenizer: Tokenizer = None
) -> Tuple[mx.array, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of strings, along with the ids
    of the most probable language tokens and the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interfere with kv-caching.

    Returns
    -------
    language_tokens : mx.array, shape = (n_audio,)
        ids of the most probable language tokens, which appears after the startoftranscript token.
    language_probs : List[Dict[str, float]], length = n_audio
        list of dictionaries containing the probability distribution over all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(
            model.is_multilingual, num_languages=model.num_languages
        )
    if (
        tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequence
    ):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform lang id"
        )

    single = mel.ndim == 2
    if single:
        mel = mel[None]

    # skip encoder forward pass if already-encoded audio features were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx, model.dims.n_audio_state):
        mel = model.encoder(mel)

    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = mx.array([[tokenizer.sot]] * n_audio)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languages; suppress all non-language tokens
    mask = mx.full(logits.shape[-1], -mx.inf, dtype=mx.float32)
    mask[list(tokenizer.all_language_tokens)] = 0.0
    logits += mask
    language_tokens = mx.argmax(logits, axis=-1)
    language_token_probs = mx.softmax(logits, axis=-1)
    language_token_probs = np.array(language_token_probs)
    language_probs = [
        {
            c: language_token_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]

    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = False  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = True  # use fp16 for most of the calculation


@dataclass(frozen=True)
class DecodingResult:
    audio_features: mx.array
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def __init__(self, model: "Whisper"):
        self.model: "Whisper" = model
        self.kv_cache = None

    def logits(self, tokens: mx.array, audio_features: mx.array) -> mx.array:
        """Perform a forward pass on the decoder and return per-token logits"""
        logits, self.kv_cache, _ = self.model.decoder(
            tokens, audio_features, kv_cache=self.kv_cache
        )
        return logits.astype(mx.float32)

    def rearrange_kv_cache(self, source_indices):
        """Update the key-value cache according to the updated beams.

        Only the self-attention cache is gathered. The cross-attention K/V is
        computed from the audio features, which ``DecodingTask.run`` repeats for
        every beam of an audio item, and a beam only ever takes over a beam of
        the same audio item, so gathering it would copy identical rows (~1.2 GB
        per step on large models at beam 5). The result is bit-identical to a
        full gather of both caches.
        """
        if source_indices != list(range(len(source_indices))):
            self.kv_cache = [
                (tree_map(lambda x: x[source_indices], self_kv), cross_kv)
                for self_kv, cross_kv in self.kv_cache
            ]

    def reset(self):
        self.kv_cache = None


class SequenceRanker:
    def rank(
        self, tokens: List[List[mx.array]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[List[int]]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : mx.array, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : mx.array, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        sum_logprobs: mx.array, shape = (n_batch)
            updated cumulative log probabilities for each sequence

        """
        raise NotImplementedError

    def finalize(
        self, tokens: mx.array, sum_logprobs: mx.array
    ) -> Tuple[Sequence[Sequence[mx.array]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : mx.array, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : mx.array, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[mx.array]], length = n_audio
            sequence of mx.arrays containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


@mx.compile
def categorical(logits, temp):
    return mx.random.categorical(logits / temp)


class BeamSearchDecoder(TokenDecoder):
    """Beam search decoder using dict-keyed sequence tracking.

    Mirrors OpenAI's original torch BeamSearchDecoder almost line-for-line.
    Each candidate sequence is stored as a full tuple of token ids in a dict,
    which simultaneously deduplicates, scores, and tracks history.

    Args:
        beam_size: Number of beams to maintain during search.
        eot: End-of-text token id.
        inference: Inference object with ``rearrange_kv_cache`` method.
        patience: Multiplier for how many finished candidates to accumulate
            before stopping. ``None`` defaults to 1.0.
    """

    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference,
        patience: float | None = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences: list[dict[tuple[int, ...], float]] | None = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        """Reset state for a new decoding run."""
        self.finished_sequences = None

    def _validate_input_shape(self, tokens: mx.array):
        """Validate that the input tokens shape is compatible with beam size."""
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

    def _initialize_state_if_needed(self, n_audio: int):
        """Initialize finished sequences state if not already done."""
        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]

    def _calculate_log_probs(self, logits: mx.array, sum_logprobs: mx.array):
        """Calculate log probabilities from logits."""
        # Log-softmax in MLX, then move to numpy for the dict loop.
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logprobs, sum_logprobs)
        logprobs_np = np.array(logprobs)
        sum_logprobs_np = np.array(sum_logprobs)
        return logprobs_np, sum_logprobs_np

    def _score_candidates_for_audio(
        self,
        i: int,
        tokens_list: list,
        logprobs_np: np.ndarray,
        sum_logprobs_np: np.ndarray,
    ) -> tuple[dict, dict]:
        """Score candidates for a single audio sample.

        Returns:
            tuple of (scores, sources) dictionaries
        """
        scores: dict[tuple[int, ...], float] = {}
        sources: dict[tuple[int, ...], int] = {}

        # Score all candidates: each beam proposes its top-(beam_size+1).
        for j in range(self.beam_size):
            idx = i * self.beam_size + j
            prefix = tokens_list[idx]
            top_k = self.beam_size + 1
            top_indices = np.argpartition(logprobs_np[idx], -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-logprobs_np[idx][top_indices])]

            for tok_idx in top_indices:
                token = int(tok_idx)
                new_logprob = float(sum_logprobs_np[idx] + logprobs_np[idx][tok_idx])
                sequence = tuple(prefix + [token])
                scores[sequence] = new_logprob
                sources[sequence] = idx

        return scores, sources

    def _select_top_beams_and_finished(
        self, scores: dict, sources: dict
    ) -> tuple[list, list, dict]:
        """Select top beams and separate finished (EOT) sequences.

        Returns:
            tuple of (next_tokens, source_indices, finished) lists/dict
        """
        next_tokens: list[list[int]] = []
        source_indices: list[int] = []
        finished: dict[tuple[int, ...], float] = {}

        # Pick the top beam_size non-EOT sequences; collect EOT sequences.
        saved = 0
        for sequence in sorted(scores, key=scores.get, reverse=True):
            if sequence[-1] == self.eot:
                finished[sequence] = scores[sequence]
            else:
                next_tokens.append(list(sequence))
                source_indices.append(sources[sequence])
                saved += 1
                if saved == self.beam_size:
                    break

        return next_tokens, source_indices, finished

    def _update_kv_cache(self, source_indices: list[int]):
        """Update the KV cache based on selected source indices."""
        self.inference.rearrange_kv_cache(source_indices)

    def _merge_finished_sequences(
        self, finished_sequences: list[dict[tuple[int, ...], float]]
    ) -> bool:
        """Merge newly finished sequences and check if decoding is complete.

        Returns:
            bool indicating if all audio samples have enough candidates
        """
        # Merge newly finished sequences into the persistent store.
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break
                previously_finished[seq] = newly_finished[seq]

        completed = all(
            len(seqs) >= self.max_candidates for seqs in self.finished_sequences
        )
        return completed

    def update(
        self,
        tokens: mx.array,
        logits: mx.array,
        sum_logprobs: mx.array,
    ) -> tuple[mx.array, bool, mx.array]:
        """Select next tokens via beam search.

        Args:
            tokens: All tokens so far, shape ``(n_batch, seq_len)``.
            logits: Decoder output logits, shape ``(n_batch, vocab_size)``.
            sum_logprobs: Cumulative log-probs per beam, shape ``(n_batch,)``.

        Returns:
            A 3-tuple ``(tokens, completed, sum_logprobs)``.
        """
        self._validate_input_shape(tokens)

        n_audio = tokens.shape[0] // self.beam_size
        self._initialize_state_if_needed(n_audio)

        logprobs_np, sum_logprobs_np = self._calculate_log_probs(logits, sum_logprobs)
        tokens_list = np.array(tokens).tolist()

        next_tokens: list[list[int]] = []
        source_indices: list[int] = []
        finished_sequences: list[dict[tuple[int, ...], float]] = []
        new_sum_logprobs: list[float] = []

        for i in range(n_audio):
            scores, sources = self._score_candidates_for_audio(
                i, tokens_list, logprobs_np, sum_logprobs_np
            )

            beam_next_tokens, beam_source_indices, finished = (
                self._select_top_beams_and_finished(scores, sources)
            )

            next_tokens.extend(beam_next_tokens)
            source_indices.extend(beam_source_indices)
            # Convert beam_next_tokens to tuples to look up scores
            new_sum_logprobs.extend([scores[tuple(seq)] for seq in beam_next_tokens])
            finished_sequences.append(finished)

        tokens = mx.array(next_tokens)
        sum_logprobs = mx.array(new_sum_logprobs)
        self._update_kv_cache(source_indices)

        completed = self._merge_finished_sequences(finished_sequences)
        return tokens, completed, sum_logprobs

    def drain_unfinished_beams(
        self, n_audio: int, sum_logprobs_np: np.ndarray, tokens_np: np.ndarray
    ):
        """Drain unfinished beams into finished sequences, sorted by score descending."""
        for i in range(n_audio):
            if len(self.finished_sequences[i]) < self.beam_size:
                for j in list(np.argsort(sum_logprobs_np[i]))[::-1]:
                    sequence = tuple(tokens_np[i, j].tolist()) + (self.eot,)
                    self.finished_sequences[i][sequence] = float(sum_logprobs_np[i][j])
                    if len(self.finished_sequences[i]) >= self.beam_size:
                        break

    def _build_sequence_output(
        self, n_audio: int, tokens_np: np.ndarray
    ) -> tuple[list[list[list[int]]], list[list[float]]]:
        """Build padded sequence output from finished sequences.

        Returns:
            tuple of (all_seqs, all_scores) where:
            - all_seqs: list of sequences per audio sample
            - all_scores: list of scores per audio sample
        """
        all_seqs: list[list[list[int]]] = []
        all_scores: list[list[float]] = []

        for i in range(n_audio):
            seqs = [list(s) for s in self.finished_sequences[i].keys()]
            scores = list(self.finished_sequences[i].values())

            if not seqs:
                seqs = [list(tokens_np[i, 0]) + [self.eot]]
                scores = [float("-inf")]

            max_len = max(len(s) for s in seqs)
            seqs = [s + [self.eot] * (max_len - len(s)) for s in seqs]
            all_seqs.append(seqs)
            all_scores.append(scores)

        return all_seqs, all_scores

    def _pad_sequences_to_global_max(
        self, all_seqs: list[list[list[int]]]
    ) -> list[list[list[int]]]:
        """Pad all sequences to the global maximum length across all audio samples."""
        global_max_len = max(len(s) for seqs in all_seqs for s in seqs)
        padded_seqs = [
            [s + [self.eot] * (global_max_len - len(s)) for s in seqs]
            for seqs in all_seqs
        ]
        return padded_seqs

    def finalize(
        self,
        tokens: mx.array,
        sum_logprobs: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Finalize beam search and return padded candidate sequences.

        Args:
            tokens: Shape ``(n_audio, n_group, seq_len)``.
            sum_logprobs: Shape ``(n_audio, n_group)``.

        Returns:
            A 2-tuple ``(tokens, sum_logprobs)`` as padded ``mx.array``s.
        """
        n_audio, n_group, seq_len = tokens.shape
        mx.eval(tokens, sum_logprobs)
        sum_logprobs_np = np.array(sum_logprobs)
        tokens_np = np.array(tokens)

        if self.finished_sequences is None:
            self.finished_sequences = [{} for _ in range(n_audio)]

        self.drain_unfinished_beams(n_audio, sum_logprobs_np, tokens_np)

        all_seqs, all_scores = self._build_sequence_output(n_audio, tokens_np)

        padded_seqs = self._pad_sequences_to_global_max(all_seqs)

        return mx.array(padded_seqs), mx.array(all_scores)


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array
    ) -> Tuple[mx.array, bool, mx.array]:
        if self.temperature == 0:
            next_tokens = logits.argmax(axis=-1)
        else:
            next_tokens = categorical(logits, self.temperature)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        current_logprobs = logprobs[mx.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        eot_mask = tokens[:, -1] == self.eot
        next_tokens = next_tokens * (1 - eot_mask) + self.eot * eot_mask
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)

        completed = mx.all(tokens[:, -1] == self.eot)
        return tokens, completed, sum_logprobs

    def finalize(self, tokens: mx.array, sum_logprobs: mx.array):
        # make sure each sequence has at least one EOT token at the end
        tokens = mx.pad(tokens, [(0, 0), (0, 0), (0, 1)], constant_values=self.eot)
        return tokens, sum_logprobs


class LogitFilter:
    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Apply any filtering or masking to logits

        Parameters
        ----------
        logits : mx.array, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : mx.array, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int, n_vocab: int):
        self.sample_begin = sample_begin
        mask = np.zeros(n_vocab, np.float32)
        mask[tokenizer.encode(" ") + [tokenizer.eot]] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        if tokens.shape[1] == self.sample_begin:
            return logits + self.mask
        return logits


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int], n_vocab: int):
        mask = np.zeros(n_vocab, np.float32)
        mask[list(suppress_tokens)] = -np.inf
        self.mask = mx.array(mask)

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        return logits + self.mask


class ApplyTimestampRules(LogitFilter):
    def __init__(
        self,
        tokenizer: Tokenizer,
        sample_begin: int,
        max_initial_timestamp_index: Optional[int],
    ):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin
        self.max_initial_timestamp_index = max_initial_timestamp_index

    def apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        mask = np.zeros(logits.shape, np.float32)
        # suppress <|notimestamps|> which is handled by without_timestamps
        if self.tokenizer.no_timestamps is not None:
            mask[:, self.tokenizer.no_timestamps] = -np.inf

        ## timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        tokens = tokens.tolist()
        for k in range(len(tokens)):
            seq = tokens[k][self.sample_begin :]
            last_was_timestamp = (
                len(seq) >= 1 and seq[-1] >= self.tokenizer.timestamp_begin
            )
            penultimate_was_timestamp = (
                len(seq) < 2 or seq[-2] >= self.tokenizer.timestamp_begin
            )

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    mask[k, self.tokenizer.timestamp_begin :] = -np.inf
                else:  # cannot be normal text tokens
                    mask[k, : self.tokenizer.eot] = -np.inf

            timestamps = [v for v in seq if v >= self.tokenizer.timestamp_begin]
            if len(timestamps) > 0:
                # timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
                # also force each segment to have a nonzero length, to prevent infinite looping
                if last_was_timestamp and not penultimate_was_timestamp:
                    last_timestamp = timestamps[-1]
                else:
                    last_timestamp = timestamps[-1] + 1
                mask[k, self.tokenizer.timestamp_begin : last_timestamp] = -np.inf

        if len(tokens[0]) == self.sample_begin:
            # suppress generating non-timestamp tokens at the beginning
            mask[:, : self.tokenizer.timestamp_begin] = -np.inf

            # apply the `max_initial_timestamp` option
            if self.max_initial_timestamp_index is not None:
                last_allowed = (
                    self.tokenizer.timestamp_begin + self.max_initial_timestamp_index
                )
                mask[:, last_allowed + 1 :] = -np.inf

        # if sum of probability over timestamps is above any other token, sample timestamp
        # decide on the masked logits, as openai/whisper and transformers do — deciding
        # on the raw logits can mask every token after a closed timestamp pair, because
        # the pair mask has already zeroed the timestamps
        mask = mx.array(mask)
        masked = logits + mask
        logprobs = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
        timestamp_logprob = logprobs[:, self.tokenizer.timestamp_begin :].logsumexp(
            axis=-1, keepdims=True
        )
        max_text_token_logprob = logprobs[:, : self.tokenizer.timestamp_begin].max(
            axis=-1, keepdims=True
        )
        mask[:, : self.tokenizer.timestamp_begin] = mx.where(
            timestamp_logprob > max_text_token_logprob,
            -mx.inf,
            mask[:, : self.tokenizer.timestamp_begin],
        )
        return logits + mask


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "Whisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        self.inference = Inference(model)

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.beam_size is not None:
            self.decoder = BeamSearchDecoder(
                options.beam_size,
                tokenizer.eot,
                self.inference,
                options.patience,
            )
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(
                SuppressBlank(self.tokenizer, self.sample_begin, model.dims.n_vocab)
            )
        if self.options.suppress_tokens:
            self.logit_filters.append(
                SuppressTokens(self._get_suppress_tokens(), model.dims.n_vocab)
            )

        if not options.without_timestamps:
            precision = CHUNK_LENGTH / model.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            if options.max_initial_timestamp:
                max_initial_timestamp_index = round(
                    self.options.max_initial_timestamp / precision
                )
            self.logit_filters.append(
                ApplyTimestampRules(
                    tokenizer, self.sample_begin, max_initial_timestamp_index
                )
            )

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: mx.array):
        if self.options.fp16:
            mel = mel.astype(mx.float16)

        if mel.shape[-2:] == (
            self.model.dims.n_audio_ctx,
            self.model.dims.n_audio_state,
        ):
            # encoded audio features are given; skip audio encoding
            audio_features = mel
        else:
            audio_features = self.model.encoder(mel)

        if audio_features.dtype != (mx.float16 if self.options.fp16 else mx.float32):
            raise TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        return audio_features

    def _detect_language(self, audio_features: mx.array, tokens: np.array):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                # write language tokens
                tokens[:, self.sot_index + 1] = np.array(lang_tokens)

        return languages, lang_probs

    def _main_loop(self, audio_features: mx.array, tokens: mx.array):
        n_batch = tokens.shape[0]
        sum_logprobs = mx.zeros(n_batch)

        def _step(inputs, audio_features, tokens, sum_logprobs):
            pre_logits = self.inference.logits(inputs, audio_features)

            # consider the logits at the last token only
            logits = pre_logits[:, -1]

            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in self.logit_filters:
                logits = logit_filter.apply(logits, tokens)

            # expand the tokens tensor with the selected next tokens
            tokens, completed, sum_logprobs = self.decoder.update(
                tokens, logits, sum_logprobs
            )
            return tokens, completed, sum_logprobs, pre_logits

        tokens, completed, sum_logprobs, pre_logits = _step(
            tokens, audio_features, tokens, sum_logprobs
        )
        if self.tokenizer.no_speech is not None:  # compute no_speech_probs
            probs_at_sot = mx.softmax(pre_logits[:, self.sot_index], axis=-1)
            no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]
        else:
            no_speech_probs = mx.full(n_batch, mx.nan)
        mx.async_eval(completed, tokens, sum_logprobs, no_speech_probs)

        for i in range(1, self.sample_len):
            inputs = tokens[:, -1:]
            if tokens.shape[-1] > self.n_ctx:
                break

            next_tokens, next_completed, next_sum_logprobs, _ = _step(
                inputs, audio_features, tokens, sum_logprobs
            )
            mx.async_eval(next_completed, next_tokens, next_sum_logprobs)
            if completed:
                break
            tokens = next_tokens
            completed = next_completed
            sum_logprobs = next_sum_logprobs

        return tokens, sum_logprobs, no_speech_probs

    def run(self, mel: mx.array) -> List[DecodingResult]:
        self.inference.reset()
        self.decoder.reset()
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel.shape[0]

        audio_features: mx.array = self._get_audio_features(mel)  # encoder forward pass
        tokens: mx.array = mx.array(self.initial_tokens)
        tokens = mx.broadcast_to(tokens, (n_audio, len(self.initial_tokens)))

        # detect language if requested, overwriting the language token
        languages, language_probs = self._detect_language(audio_features, tokens)
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, language=language, language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]

        # repeat tokens by the group size, for beam search or best-of-n sampling
        if self.n_group > 1:
            tokens = tokens[:, None, :]
            tokens = mx.broadcast_to(
                tokens, [n_audio, self.n_group, len(self.initial_tokens)]
            )
            tokens = tokens.reshape((n_audio * self.n_group, len(self.initial_tokens)))
            # Expand audio_features to match tokens batch dim so that the decoder's
            # cross-attention KV cache is initialised with the correct batch size.
            # Without this, the KV cache has shape (n_audio, ...) while self-attention
            # queries have shape (n_audio * n_group, ...), causing NaN logits from the
            # very first step.  mx.repeat interleaves each row n_group times, which is
            # the layout expected by rearrange_kv_cache and the [::n_group] slice below.
            audio_features = mx.repeat(audio_features, self.n_group, axis=0)

        # call the main sampling loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop(audio_features, tokens)

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(tokens, sum_logprobs)
        tokens = tokens[..., self.sample_begin :]

        # eval and convert to list
        mx.eval(tokens, sum_logprobs, no_speech_probs)
        tokens = tokens.tolist()
        sum_logprobs = sum_logprobs.tolist()
        no_speech_probs = no_speech_probs.tolist()
        tokens = [[t[: t.index(tokenizer.eot)] for t in s] for s in tokens]

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i] for i, t in zip(selected, tokens)]
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (
            texts,
            languages,
            tokens,
            audio_features,
            avg_logprobs,
            no_speech_probs,
        )
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        return [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(text),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]


def decode(
    model: "Whisper",
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: mx.array, shape = (80, 3000) or (*, 80, 3000)
        An array containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if single := mel.ndim == 2:
        mel = mel[None]

    if kwargs:
        options = replace(options, **kwargs)

    result = DecodingTask(model, options).run(mel)
    return result[0] if single else result
