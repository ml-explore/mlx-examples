# Copyright © 2024 Apple Inc.

import argparse
import os
import pathlib
import traceback
import warnings

from . import audio
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from .transcribe import transcribe
from .writers import get_writer


def build_parser():
    def optional_int(string):
        return None if string == "None" else int(string)

    def optional_float(string):
        return None if string == "None" else float(string)

    def str2bool(string):
        str2val = {"True": True, "False": False}
        if string in str2val:
            return str2val[string]
        else:
            raise ValueError(f"Expected one of {set(str2val.keys())}, got {string}")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")

    parser.add_argument(
        "--model",
        default="mlx-community/whisper-tiny",
        type=str,
        help="The model directory or hugging face repo",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "The name of transcription/translation output files before "
            "--output-format extensions"
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="Directory to save the outputs",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        default="txt",
        choices=["txt", "vtt", "srt", "tsv", "json", "all"],
        help="Format of the output file",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="Whether to print out progress and debug messages",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Perform speech recognition ('transcribe') or speech translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="Language spoken in the audio, specify None to auto-detect",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="Temperature for sampling"
    )
    parser.add_argument(
        "--best-of",
        type=optional_int,
        default=5,
        help="Number of candidates when sampling with non-zero temperature",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help="Optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="Optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default.",
    )
    parser.add_argument(
        "--suppress-tokens",
        type=str,
        default="-1",
        help="Comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Optional text to provide as a prompt for the first window.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        type=str2bool,
        default=True,
        help="If True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=True,
        help="Whether to perform inference in fp16",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=optional_float,
        default=2.4,
        help="if the gzip compression ratio is higher than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=optional_float,
        default=-1.0,
        help="If the average log probability is lower than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=optional_float,
        default=0.6,
        help="If the probability of the token is higher than this value the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    parser.add_argument(
        "--word-timestamps",
        type=str2bool,
        default=False,
        help="Extract word-level timestamps and refine the results based on them",
    )
    parser.add_argument(
        "--prepend-punctuations",
        type=str,
        default="\"'“¿([{-",
        help="If word-timestamps is True, merge these punctuation symbols with the next word",
    )
    parser.add_argument(
        "--append-punctuations",
        type=str,
        default="\"'.。,，!！?？:：”)]}、",
        help="If word_timestamps is True, merge these punctuation symbols with the previous word",
    )
    parser.add_argument(
        "--highlight-words",
        type=str2bool,
        default=False,
        help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt",
    )
    parser.add_argument(
        "--max-line-width",
        type=int,
        default=None,
        help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line",
    )
    parser.add_argument(
        "--max-line-count",
        type=int,
        default=None,
        help="(requires --word_timestamps True) the maximum number of lines in a segment",
    )
    parser.add_argument(
        "--max-words-per-line",
        type=int,
        default=None,
        help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=optional_float,
        help="(requires --word_timestamps True) skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected",
    )
    parser.add_argument(
        "--clip-timestamps",
        type=str,
        default="0",
        help="Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file",
    )
    return parser


def main():
    parser = build_parser()
    args = vars(parser.parse_args())
    if args["verbose"] is True:
        print(f"Args: {args}")

    path_or_hf_repo: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    output_name: str = args.pop("output_name")
    os.makedirs(output_dir, exist_ok=True)

    writer = get_writer(output_format, output_dir)
    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    writer_args = {arg: args.pop(arg) for arg in word_options}
    if not args["word_timestamps"]:
        for k, v in writer_args.items():
            if v:
                argop = k.replace("_", "-")
                parser.error(f"--{argop} requires --word-timestamps True")
    if writer_args["max_line_count"] and not writer_args["max_line_width"]:
        warnings.warn("--max-line-count has no effect without --max-line-width")
    if writer_args["max_words_per_line"] and writer_args["max_line_width"]:
        warnings.warn("--max-words-per-line has no effect with --max-line-width")

    for audio_obj in args.pop("audio"):
        if audio_obj == "-":
            # receive the contents from stdin rather than read a file
            audio_obj = audio.load_audio(from_stdin=True)

            output_name = output_name or "content"
        else:
            output_name = output_name or pathlib.Path(audio_obj).stem
        try:
            result = transcribe(
                audio_obj,
                path_or_hf_repo=path_or_hf_repo,
                **args,
            )
            writer(result, output_name, **writer_args)
        except Exception as e:
            traceback.print_exc()
            print(f"Skipping {audio_obj} due to {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
