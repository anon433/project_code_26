import os
from pathlib import Path

from setuptools import find_packages, setup

repo_root = Path(__file__).resolve().parent
requirements_path = Path(
    os.environ.get("SPARSE_UNLEARN_REQUIREMENTS_FILE", "requirements.txt")
)
if not requirements_path.is_absolute():
    requirements_path = repo_root / requirements_path
requirements = [
    line
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines()
    if (line := raw_line.strip()) and not line.startswith("#")
]

setup(
    name="sparse-unlearning-reproduction",
    version="0.1.0",
    description="A library for machine unlearning in LLMs.",
    long_description=(repo_root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,  # Uses requirements.txt
    extras_require={
        "reproduction": [
            "lm-eval==0.4.8",
            "fastargs==1.2.0",
            "terminaltables==3.1.10",
            "peft==0.13.2",
            "sentencepiece==0.2.2",
            "bitsandbytes==0.49.2",
        ],
        "distributed": ["deepspeed==0.15.4"],
        "quantization": ["bitsandbytes==0.49.2"],
        "lm-eval": [
            "lm-eval==0.4.8",
            "peft==0.13.2",
        ],  # Install using `pip install .[lm-eval]`
        "dev": [
            "pre-commit==4.0.1",
            "ruff==0.6.9",
        ],  # Install using `pip install .[dev]`
    },
    python_requires=">=3.11,<3.12",
)
