# Copyright © 2024 Apple Inc.

import os
import sys
from pathlib import Path

from setuptools import find_namespace_packages, setup

# 获取当前文件的父目录（项目根目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(ROOT_DIR, "mlx_flux")

# 定义依赖列表
requirements = []
if os.path.exists(os.path.join(ROOT_DIR, "requirements.txt")):
    with open(os.path.join(ROOT_DIR, "requirements.txt")) as fid:
        requirements = [l.strip() for l in fid.readlines() if l.strip()]

# 添加包路径
sys.path.append(package_dir)

from _version import __version__

try:
    with open(os.path.join(ROOT_DIR, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "FLUX.1 on Apple silicon with MLX and the Hugging Face Hub"

setup(
    name="mlx-flux",
    version=__version__,
    description="FLUX.1 on Apple silicon with MLX and the Hugging Face Hub",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author_email="mlx@group.apple.com",
    author="MLX Contributors",
    url="https://github.com/ml-explore/mlx-examples",
    license="MIT",
    install_requires=requirements,
    # Package configuration
    packages=find_namespace_packages(
        include=["mlx_flux", "mlx_flux.*"]
    ),  # 明确指定包含的包
    package_data={
        "mlx_flux": ["*.py"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            # generate images
            "mlx_flux.generate = mlx_flux.txt2image:main",
            "mlx_flux.txt2image = mlx_flux.txt2image:main",
            # fine-tuning model
            "mlx_flux.lora = mlx_flux.dreambooth:main",
            "mlx_flux.dreambooth = mlx_flux.dreambooth:main",
        ]
    },
)
