import argparse

from PIL import Image

from stable_fast_3d import StableFast3D


def main():
    parser = argparse.ArgumentParser(
        description="Single-image to textured 3D mesh (SF3D, native MLX)."
    )
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "output",
        nargs="?",
        default="output.glb",
        help="Output GLB path (default: output.glb).",
    )
    parser.add_argument("--foreground-ratio", type=float, default=0.85)
    parser.add_argument(
        "--no-remove-bg",
        action="store_true",
        help="Skip background removal (image already has a clean alpha/background).",
    )
    parser.add_argument("--bake-resolution", type=int, default=512)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model = StableFast3D()
    image = Image.open(args.image)
    out = model.run(
        image,
        args.output,
        foreground_ratio=args.foreground_ratio,
        remove_bg=not args.no_remove_bg,
        bake_resolution=args.bake_resolution,
        verbose=args.verbose,
    )
    print(out)


if __name__ == "__main__":
    main()
