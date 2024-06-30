import argparse
import os
import traceback
import warnings

from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE
from .transcribe import transcribe
from .utils import get_writer, optional_float, optional_int, str2bool


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio", nargs="+", type=str, help="audio file(s) to transcribe"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/whisper-tiny",
        type=str,
        help="the path to save model files, or the hugging face repo id to use",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=".",
        help="directory to save the outputs",
    )
    parser.add_argument(
        "--output-format",
        "-f",
        type=str,
        default="all",
        choices=["txt", "vtt", "srt", "tsv", "json", "all"],
        help="format of the output file; if not specified, all available formats will be produced",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=True,
        help="whether to print out the progress and debug messages",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=sorted(LANGUAGES.keys())
        + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]),
        help="language spoken in the audio, specify None to perform language detection",
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature to use for sampling"
    )
    parser.add_argument(
        "--best-of",
        type=optional_int,
        default=5,
        help="number of candidates when sampling with non-zero temperature",
    )
    parser.add_argument(
        "--patience",
        type=float,
        default=None,
        help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=None,
        help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default",
    )
    parser.add_argument(
        "--suppress-tokens",
        type=str,
        default="-1",
        help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="optional text to provide as a prompt for the first window.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        type=str2bool,
        default=True,
        help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop",
    )
    parser.add_argument(
        "--fp16",
        type=str2bool,
        default=True,
        help="whether to perform inference in fp16; True by default",
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
        help="if the average log probability is lower than this value, treat the decoding as failed",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=optional_float,
        default=0.6,
        help="if the probability of the  token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence",
    )
    parser.add_argument(
        "--word-timestamps",
        type=str2bool,
        default=False,
        help="(experimental) extract word-level timestamps and refine the results based on them",
    )
    parser.add_argument(
        "--prepend-punctuations",
        type=str,
        default="\"'“¿([{-",
        help="if word_timestamps is True, merge these punctuation symbols with the next word",
    )
    parser.add_argument(
        "--append-punctuations",
        type=str,
        default="\"'.。,，!！?？:：”)]}、",
        help="if word_timestamps is True, merge these punctuation symbols with the previous word",
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
        "--clip-timestamps",
        type=str,
        default="0",
        help="comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file",
    )
    parser.add_argument(
        "--hallucination-silence-threshold",
        type=optional_float,
        help="(requires --word_timestamps True) skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args().__dict__
    print(f"Raw Args: {args}")
    print(f"-----")
    # Convert hyphenated args to underscore
    args = {k.replace("-", "_"): v for k, v in args.items()}

    if args["verbose"] is True:
        print(f"Args: {args}")

    path_or_hf_repo: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    os.makedirs(output_dir, exist_ok=True)

    temperature = args.pop("temperature")

    writer = get_writer(output_format, output_dir)
    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    for audio_path in args.pop("audio"):
        try:
            result = transcribe(
                audio_path,
                path_or_hf_repo=path_or_hf_repo,
                temperature=temperature,
                **args,
            )
            writer(result, audio_path, **writer_args)
        except Exception as e:
            traceback.print_exc()
            print(f"Skipping {audio_path} due to {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    main()
