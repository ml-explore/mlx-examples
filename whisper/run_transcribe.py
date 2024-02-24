from whisper import transcribe


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="The path to audio file to transcribe",
    )
    parser.add_argument(
        "--model",
        default="mlx_models/tiny_fp16",
        help="The path to the local model directory or Hugging Face repo.",
    )
    return parser


if __name__ == "__main__":
    """
    python transcribe.py \
      --audio ./some/path/to/some-audio-file.m4a.or.wav \
      --model ./some_mlx_model/tiny_fp16
    """
    parser = build_parser()
    args = parser.parse_args()

    model = args.model
    audio = args.audio

    result = transcribe(audio, path_or_hf_repo=model)
    print(result["text"])
