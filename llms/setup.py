# Copyright © 2024 Apple Inc.

import sys
from pathlib import Path

from setuptools import setup

package_dir = Path(__file__).parent / "mlx_lm"
with open(package_dir / "requirements.txt") as fid:
    requirements = [l.strip() for l in fid.readlines()]

sys.path.append(str(package_dir))
from version import __version__

setup(
    name="mlx-lm",
    version=__version__,
    description="LLMs on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-examples",
    license="MIT",
    install_requires=requirements,
    packages=["mlx_lm", "mlx_lm.models", "mlx_lm.tuner"],
    python_requires=">=3.8",
    extras_require={
        "testing": ["datasets"],
    },
    entry_points={
        "console_scripts": [
            "mlx_lm.convert = mlx_lm.convert:main",
            "mlx_lm.fuse = mlx_lm.fuse:main",
            "mlx_lm.generate = mlx_lm.generate:main",
            "mlx_lm.lora = mlx_lm.lora:main",
            "mlx_lm.merge = mlx_lm.merge:main",
            "mlx_lm.server = mlx_lm.server:main",
            "mlx_lm.manage = mlx_lm.manage:main",
        ]
    },
)
