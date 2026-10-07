"""Build a source-only anonymous ZIP with normalized metadata."""

import argparse
import os
import tempfile
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT_FILES = {
    "README.md",
    "LICENSE",
    "Makefile",
    "setup.py",
    "requirements.txt",
    ".gitignore",
    "ruff.toml",
    "mask_search.sh",
    "stage2.sh",
}


def members(root):
    root = Path(root).resolve()
    selected = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        parts = relative.parts
        included = (
            relative.as_posix() in ROOT_FILES
            or relative.as_posix() == "docs/reproduction.md"
            or (parts[0] in {"src", "tests"} and path.suffix == ".py")
            or (parts[0] == "configs" and path.suffix == ".yaml")
            or (
                parts[0] == "third_party"
                and (path.suffix == ".py" or path.name == "LICENSE")
            )
        )
        if not included or "__pycache__" in parts:
            continue
        if any(
            parent.is_symlink()
            for parent in [path, *path.parents]
            if parent != root.parent
        ):
            raise ValueError(f"Archive symlink is forbidden: {relative}")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe archive path: {relative}")
        selected.append(relative.as_posix())
    return selected


def scan(root, private_terms=()):
    root = Path(root)
    patterns = {
        "machine path": re.compile(r"/(?:root|home|Users)/[\w.-]+"),
        "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
        "access token": re.compile(r"\b(?:hf_|ghp_)[A-Za-z0-9]{20,}\b"),
    }
    findings = []
    for name in members(root):
        content = (root / name).read_text(encoding="utf-8")
        for label, pattern in patterns.items():
            for line in content.splitlines():
                if (
                    label == "email"
                    and name.endswith(".yaml")
                    and re.match(r"\s*-\s*(?:override\s+)?[\w./]+@[\w.]+:", line)
                ):
                    line = re.sub(
                        r"\s*-\s*(?:override\s+)?[\w./]+@[\w.]+:", "", line, count=1
                    )
                if pattern.search(line):
                    findings.append({"file": name, "reason": label})
                    break
        if any(
            term and term.casefold() in (name + "\n" + content).casefold()
            for term in private_terms
        ):
            findings.append({"file": name, "reason": "private identity term"})
    return findings


def build_archive(root, output, *, private_terms=()):
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.resolve().is_relative_to(root):
        raise ValueError("Archive output must be outside the source directory")
    findings = scan(root, private_terms)
    if findings:
        raise ValueError(f"Anonymous scan failed: {findings}")
    names = members(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(
        prefix=".source-archive-", suffix=".zip", dir=output.parent.resolve()
    )
    os.close(handle)
    temporary = Path(name)
    inventory = {}
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in names:
            payload = (root / name).read_bytes()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
            inventory[name] = hashlib.sha256(payload).hexdigest()
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None or archive.namelist() != names:
            raise ValueError("Archive membership/integrity failure")
        for name, digest in inventory.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError(f"Archive content mismatch: {name}")
    temporary.replace(output)
    return dict(sha256=hashlib.sha256(output.read_bytes()).hexdigest(), files=inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--private-terms",
        type=Path,
        help="Local JSON list; kept outside the source/archive",
    )
    args = parser.parse_args()
    terms = json.loads(args.private_terms.read_text()) if args.private_terms else []
    result = build_archive(
        Path(__file__).resolve().parents[2], args.output, private_terms=terms
    )
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent.resolve(), delete=False
    ) as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
        manifest = Path(handle.name)
    manifest.replace(args.output.with_suffix(".manifest.json"))
    print(
        json.dumps(
            {
                "archive": str(args.output),
                "sha256": result["sha256"],
                "files": len(result["files"]),
            }
        )
    )


if __name__ == "__main__":
    main()
