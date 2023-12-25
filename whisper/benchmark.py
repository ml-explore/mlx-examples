# Copyright © 2023 Apple Inc.

import sys
import time

import mlx.core as mx

from whisper import audio, decoding, load_models, transcribe

audio_file = "whisper/assets/ls_test.flac"


def timer(fn, *args):
    for _ in range(5):
        fn(*args)

    num_its = 10

    tic = time.perf_counter()
    for _ in range(num_its):
        fn(*args)
    toc = time.perf_counter()
    return (toc - tic) / num_its


def feats():
    data = audio.load_audio(audio_file)
    data = audio.pad_or_trim(data)
    mels = audio.log_mel_spectrogram(data)
    mx.eval(mels)
    return mels


def model_forward(model, mels, tokens):
    logits = model(mels, tokens)
    mx.eval(logits)
    return logits


def decode(model, mels):
    return decoding.decode(model, mels)


def everything():
    return transcribe(audio_file)


if __name__ == "__main__":

    # get command line arguments without 3rd party libraries
    # the command line argument to benchmark all models is "all"
    models = ["tiny"]
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            models = ["tiny", "small", "medium", "large"]

    feat_time = timer(feats)
    print(f"\nFeature time {feat_time:.3f}")
    mels = feats()[None].astype(mx.float16)

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
        model_forward_time = timer(model_forward, model, mels, tokens)
        print(f"Model forward time {model_forward_time:.3f}")
        decode_time = timer(decode, model, mels)
        print(f"Decode time {decode_time:.3f}")
        everything_time = timer(everything)
        print(f"Everything time {everything_time:.3f}")
        print(f"\n{'-----' * 10}\n")
