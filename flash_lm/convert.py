import argparse
import json
import shutil
from pathlib import Path

import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a model for use with mlx-lm.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        type=str,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--save-dir",
        default="mlx_lm_checkpoint",
        type=str,
        help="Location to save the mlx_lm ready model",
    )
    args = parser.parse_args()

    tokenizer = utils.load_tokenizer()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save tokenizer
    tokenizer.save_pretrained(save_dir)

    checkpoint_dir = Path(args.checkpoint_dir)
    config = utils.load_config(checkpoint_dir).model
    config["model_file"] = f"{config['model_type']}.py"
    config = dict(sorted(config.items()))
    with open(save_dir / "config.json", "w") as fid:
        json.dump(config, fid, indent=4)

    for file in [
        "models/transformer.py",
        checkpoint_dir / "model.safetensors",
    ]:
        dst_path = save_dir / Path(file).name
        shutil.copy(file, dst_path)
