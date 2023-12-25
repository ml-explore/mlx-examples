# Copyright © 2023 Apple Inc.

import sys
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult
from .load_models import load_model
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer


def _format_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class ModelHolder:
    model = None
    model_name = None

    @classmethod
    def get_model(cls, model: str, dtype: mx.Dtype):
        if cls.model is None or model != cls.model_name:
            cls.model = load_model(model, dtype=dtype)
            cls.model_name = model
        return cls.model


def transcribe(
    audio: Union[str, np.ndarray, mx.array],
    *,
    model: str = "tiny",
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array]
        The path to the audio file to open, or the audio waveform

    model: str
        The Whisper model. Can be any of ["tiny", "base", "small", "medium", "large"].
        Default is "tiny".

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    dtype = mx.float16 if decode_options.get("fp16", True) else mx.float32
    model = ModelHolder.get_model(model, dtype)

    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-2] - N_FRAMES

    if verbose:
        system_encoding = sys.getdefaultencoding()
        if system_encoding != "utf-8":
            make_safe = lambda x: x.encode(system_encoding, errors="replace").decode(
                system_encoding
            )
        else:
            make_safe = lambda x: x

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. "
                    "Use the `language` decoding option to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )

    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )

    def decode_with_fallback(segment: mx.array) -> DecodingResult:
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return decode_result

    seek = 0
    input_stride = N_FRAMES // model.dims.n_audio_ctx  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []

    def new_segment(
        *, start: float, end: float, tokens: mx.array, result: DecodingResult
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }

    # show the progress bar when verbose is False (if True, transcribed text will be printed)
    with tqdm.tqdm(
        total=content_frames, unit="frames", disable=verbose is not False
    ) as pbar:
        last_speech_timestamp = 0.0
        while seek < content_frames:
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            mel_segment = mel[seek : seek + N_FRAMES]
            segment_size = min(N_FRAMES, content_frames - seek)
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=-2).astype(dtype)

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(mel_segment)
            tokens = np.array(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if (
                    logprob_threshold is not None
                    and result.avg_logprob > logprob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment_size  # fast-forward to the next segment boundary
                    continue

            previous_seek = seek
            current_segments = []

            timestamp_tokens = tokens >= tokenizer.timestamp_begin
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = np.where(
                np.logical_and(timestamp_tokens[:-1], timestamp_tokens[1:])
            )[0]
            consecutive += 1
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[timestamp_tokens.nonzero()[0]]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                seek += segment_size

            if verbose:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{_format_timestamp(start)} --> {_format_timestamp(end)}] {text}"
                    print(make_safe(line))

            # if a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(
                        current_segments, start=len(all_segments)
                    )
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if not condition_on_previous_text or result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # update progress bar
            pbar.update(min(content_frames, seek) - previous_seek)

    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )
