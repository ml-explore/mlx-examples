# Copyright © 2024 Apple Inc.

import sys
from pathlib import Path

from setuptools import find_packages, setup

package_dir = Path(__file__).parent / "mlx_whisper"

with open(package_dir / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))

from version import __version__

setup(
    name="mlx-whisper",
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
    include_package_data=True,
    python_requires=">=3.8",
)
