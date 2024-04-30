import sys
from pathlib import Path

from setuptools import find_packages, setup

package_dir = Path(__file__).parent
print(f"{package_dir}")

with open(package_dir / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

package_dir = Path(__file__).parent / "mlx_whisper"
sys.path.append(str(package_dir))

from version import __version__

setup(
    name="mlx_whisper",
    version=__version__,
    description="OpenAI Whisper on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-examples",
    license="MIT",
    install_requires=requirements,
    packages=find_packages(),
    package_data={
        'mlx_whisper': [
            'assets/mel_filters.npz',
            'assets/multilingual.tiktoken',
            'assets/gpt2.tiktoken',
        ],
    },
    python_requires=">=3.8",
)
