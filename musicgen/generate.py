# Copyright © 2024 Apple Inc.

import argparse

from utils import save_audio

from musicgen import MusicGen


def main(text: str, output_path: str, model_name: str, max_steps: int):
    model = MusicGen.from_pretrained(model_name)
    audio = model.generate(text, max_steps=max_steps)
    save_audio(output_path, audio, model.sampling_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, default="facebook/musicgen-medium")
    parser.add_argument("--text", required=False, default="happy rock")
    parser.add_argument("--output-path", required=False, default="0.wav")
    parser.add_argument("--max-steps", required=False, default=500, type=int)
    args = parser.parse_args()
    main(args.text, args.output_path, args.model, args.max_steps)
