"""
Fetches and caches the two pieces of upstream Stability AI data this example
needs at runtime: the model checkpoint (converted from PyTorch to an
MLX-loadable safetensors file, cached under ~/.cache/mlx-stable-fast-3d/) and
the fixed tetrahedra grid used by isosurface extraction (downloaded once from
the upstream repo, not shipped in this example - see NOTICE.md).

The model itself is gated on Hugging Face; the caller must have already run
`huggingface-cli login` with an account that's been granted access at
https://huggingface.co/stabilityai/stable-fast-3d.
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

CACHE_DIR = Path.home() / ".cache" / "mlx-stable-fast-3d"

TETS_URL = (
    "https://raw.githubusercontent.com/Stability-AI/stable-fast-3d/main/"
    "load/tets/160_tets.npz"
)


def get_weights_path() -> Path:
    """Returns a local path to weights_mlx.safetensors, converting from the
    original PyTorch checkpoint (downloaded via huggingface_hub) on first
    use. Cached thereafter."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "weights_mlx.safetensors"
    if out_path.exists():
        return out_path

    from safetensors import safe_open
    from safetensors.numpy import save_file

    ckpt_path = hf_hub_download(
        repo_id="stabilityai/stable-fast-3d", filename="model.safetensors"
    )
    arrays = {}
    with safe_open(ckpt_path, framework="numpy") as f:
        for key in f.keys():
            arrays[key] = f.get_tensor(key)
    save_file(arrays, str(out_path))
    return out_path


def get_tets_path() -> Path:
    """Returns a local path to 160_tets.npz, downloading it from the
    upstream stable-fast-3d repo on first use. This is a fixed geometric
    grid definition (not a trained weight), authored by Stability AI and
    covered by the same license as the rest of this example - see
    NOTICE.md."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / "160_tets.npz"
    if out_path.exists():
        return out_path

    import urllib.request

    urllib.request.urlretrieve(TETS_URL, out_path)
    return out_path
