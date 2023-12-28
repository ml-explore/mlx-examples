# Copyright © 2023 Apple Inc.
import argparse
import sys
import time

import mlx.core as mx

from whisper import audio, decoding, load_models, transcribe

audio_file = "whisper/assets/ls_test.flac"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmark script.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Use all available models, i.e. tiny,small,medium,large-v3",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        help="Specify models as a comma-separated list (e.g., tiny,small,medium)",
    )
    return parser.parse_args()


def timer(fn, *args):
    for _ in range(5):
        fn(*args)

    num_its = 10

    tic = time.perf_counter()
    for _ in range(num_its):
        fn(*args)
    toc = time.perf_counter()
    return (toc - tic) / num_its


def feats(n_mels: int = 80):
    data = audio.load_audio(audio_file)
    data = audio.pad_or_trim(data)
    mels = audio.log_mel_spectrogram(data, n_mels)
    mx.eval(mels)
    return mels


def model_forward(model, mels, tokens):
    logits = model(mels, tokens)
    mx.eval(logits)
    return logits


def decode(model, mels):
    return decoding.decode(model, mels)


def everything(model_name):
    return transcribe(audio_file, model=model_name)


if __name__ == "__main__":
    args = parse_arguments()
    if args.all:
        models = ["tiny", "small", "medium", "large-v3"]
    elif args.models:
        models = args.models.split(",")
    else:
        models = ["tiny"]

    print("Selected models:", models)

    feat_time = timer(feats)
    print(f"\nFeature time {feat_time:.3f}")

    for model_name in models:
        print(f"\nModel: {model_name.upper()}")
        tokens = mx.array(
            [
                50364,
                1396,
                264,
                665,
                5133,
                23109,
                25462,
                264,
                6582,
                293,
                750,
                632,
                42841,
                292,
                370,
                938,
                294,
                4054,
                293,
                12653,
                356,
                50620,
                50620,
                23563,
                322,
                3312,
                13,
                50680,
            ],
            mx.int32,
        )[None]
        model = load_models.load_model(f"{model_name}", dtype=mx.float16)
        mels = feats(model.dims.n_mels)[None].astype(mx.float16)
        model_forward_time = timer(model_forward, model, mels, tokens)
        print(f"Model forward time {model_forward_time:.3f}")
        decode_time = timer(decode, model, mels)
        print(f"Decode time {decode_time:.3f}")
        everything_time = timer(everything, model_name)
        print(f"Everything time {everything_time:.3f}")
        print(f"\n{'-----' * 10}\n")
