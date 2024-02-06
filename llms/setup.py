import sys
from pathlib import Path

import pkg_resources
from setuptools import setup

with open(Path(__file__).parent / "mlx_lm/requirements.txt") as fid:
    requirements = [str(r) for r in pkg_resources.parse_requirements(fid)]
setup(
    name="mlx-lm",
    version="0.0.8",
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
)
