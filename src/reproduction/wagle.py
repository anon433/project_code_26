"""Run the vendored WAGLE algorithm with the required balanced model map."""

import runpy
from pathlib import Path


def balanced_loader(loader):
    def load(*args, **kwargs):
        kwargs["device_map"] = "balanced"
        return loader(*args, **kwargs)

    return load


def main():
    from transformers import AutoModelForCausalLM

    original = AutoModelForCausalLM.from_pretrained
    # The core currently hardcodes auto. Override only model placement in this
    # dedicated child process; scoring, masks and validation remain vendored.
    AutoModelForCausalLM.from_pretrained = balanced_loader(original)
    try:
        script = (
            Path(__file__).resolve().parents[2]
            / "third_party/WAGLE/scripts/run_fair_wmdp.py"
        )
        runpy.run_path(str(script), run_name="__main__")
    finally:
        AutoModelForCausalLM.from_pretrained = original


if __name__ == "__main__":
    main()
