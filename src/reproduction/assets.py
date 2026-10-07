"""Pinned Hugging Face assets and deterministic paper input preparation.

Paths use the caller's work directory and the standard Hugging Face cache.
No campaign script, mirror, proxy, or machine-specific path is required.
"""

from __future__ import annotations

import fcntl
import hashlib
import random
import re
from pathlib import Path

import torch

from .io import json_hash, read_json, sha256_file, write_json
from .config import contract_hash, load_contract
from .traces import (
    build_payload,
    consumption_hash,
    muse_plan,
    publish_stream,
    validate_stream,
    wmdp_plan,
)


def _identity(spec: dict) -> dict:
    revision = spec["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Asset revision must be a full pinned commit")
    return {
        "repo": spec["repo"],
        "repo_type": spec.get("repo_type", "model"),
        "revision": revision,
    }


def resolve_snapshot(
    spec: dict, cache_dir: Path | None = None, *, local_files_only: bool = True
) -> dict:
    from huggingface_hub import snapshot_download

    identity = _identity(spec)
    try:
        path = Path(
            snapshot_download(
                repo_id=identity["repo"],
                repo_type=identity["repo_type"],
                revision=identity["revision"],
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
    except Exception as error:
        raise FileNotFoundError(
            f"Cannot resolve pinned asset {identity['repo']}@{identity['revision']}: {error}"
        ) from error
    if path.name != identity["revision"]:
        raise ValueError("Resolved snapshot revision differs from pin")
    files = {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }
    if not files:
        raise ValueError("Empty asset snapshot")
    record = {**identity, "path": str(path.absolute()), "files": files}
    for file in path.glob("*.safetensors"):
        from safetensors import safe_open

        try:
            with safe_open(file, framework="pt") as handle:
                if not handle.keys():
                    raise ValueError("Empty tensors")
        except Exception as error:
            raise ValueError(f"Corrupt tensor snapshot: {file}") from error
    validate_snapshot(record, spec)
    return record


def validate_snapshot(record: dict, spec: dict) -> None:
    identity = _identity(spec)
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError("Asset repo/revision differs from contract")
    path = Path(record["path"])
    if path.name != identity["revision"] or not record.get("files"):
        raise ValueError("Invalid snapshot revision/path")
    inventory = {
        str(item.relative_to(path)) for item in path.rglob("*") if item.is_file()
    }
    if inventory != set(record["files"]):
        raise ValueError("Snapshot file inventory changed")
    for name, digest in record["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe snapshot file path")
        file = path / name
        if not file.is_file() or sha256_file(file) != digest:
            raise ValueError(f"Asset content changed: {file}")
        if file.is_symlink():
            blob_name = file.resolve().name
            if re.fullmatch(r"[0-9a-f]{64}", blob_name):
                observed = digest
            elif re.fullmatch(r"[0-9a-f]{40}", blob_name):
                hasher = hashlib.sha1(f"blob {file.stat().st_size}\0".encode())
                with file.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 << 20), b""):
                        hasher.update(chunk)
                observed = hasher.hexdigest()
            else:
                raise ValueError(f"Unrecognized immutable HF blob address: {file}")
            if observed != blob_name:
                raise ValueError(f"HF blob content/address mismatch: {file}")


def publish_assets(path: Path, payload: dict) -> dict:
    record = {key: value for key, value in payload.items() if key != "manifest_hash"}
    record["manifest_hash"] = json_hash(record)
    write_json(path, record)
    return record


def read_assets(path: Path, contract_digest: str) -> dict:
    record = read_json(path)
    unsigned = {key: value for key, value in record.items() if key != "manifest_hash"}
    if record.get("manifest_hash") != json_hash(unsigned):
        raise ValueError("Asset manifest hash mismatch")
    if record.get("contract_hash") != contract_digest:
        raise ValueError("Asset manifest contract drift")
    return record


def validate_evaluation_snapshot(
    record: dict, spec: dict, *, include_raw=False
) -> dict:
    """Reject partial Hub caches even when their present-file hashes are valid."""
    import pyarrow.parquet as parquet
    import yaml

    validate_snapshot(record, spec)
    root = Path(record["path"])
    card = (root / "README.md").read_text(encoding="utf-8")
    if not card.startswith("---\n"):
        raise ValueError("Evaluation snapshot has no dataset configuration metadata")
    metadata = yaml.safe_load(card.split("---", 2)[1])
    configs = metadata.get("configs", [])
    if not configs:
        raise ValueError("Evaluation snapshot has no dataset configurations")
    info = {item["config_name"]: item for item in metadata.get("dataset_info", [])}
    checked, layouts = {}, {}
    muse_columns = {
        "knowmem": {"question", "answer"},
        "privleak": {"text"},
        "verbmem": {"prompt", "gt"},
    }
    if include_raw:
        muse_columns["raw"] = {"text"}
    muse = spec["repo"].startswith("muse-bench/")
    for config in configs:
        if muse and config["config_name"] not in muse_columns:
            continue
        # lm-eval's MMLU group loads the individual subject configurations; its
        # separate `all` aggregate is unused and need not be downloaded.
        if spec["repo"] == "hails/mmlu_no_train" and config["config_name"] == "all":
            continue
        layouts[config["config_name"]] = {}
        splits = config.get("data_files", [])
        if not splits:
            raise ValueError("Evaluation configuration has no parquet splits")
        for split in splits:
            if (
                muse
                and config["config_name"] == "raw"
                and split["split"] not in {"forget", "retain1"}
            ):
                continue
            patterns = split["path"]
            patterns = [patterns] if isinstance(patterns, str) else patterns
            paths = set()
            for pattern in patterns:
                if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                    raise ValueError("Unsafe evaluation parquet path")
                matches = set(root.glob(pattern))
                if not matches:
                    raise ValueError(f"Missing evaluation parquet files: {pattern}")
                paths.update(matches)
            for path in paths:
                if (
                    muse
                    and config["config_name"] == "raw"
                    and (
                        path.parent != root / "raw"
                        or not re.fullmatch(
                            rf"{split['split']}-\d{{5}}-of-\d{{5}}\.parquet", path.name
                        )
                    )
                ):
                    raise ValueError(
                        "MUSE raw split points outside its pinned parquet shard names"
                    )
                if (
                    str(path.relative_to(root)) not in record["files"]
                    or path.suffix != ".parquet"
                ):
                    raise ValueError(
                        "Evaluation parquet inventory differs from manifest"
                    )
                if path not in checked:
                    table = parquet.ParquetFile(path)
                    if (
                        not (
                            muse_columns[config["config_name"]]
                            if muse
                            else {"question", "choices", "answer"}
                        ).issubset(table.schema_arrow.names)
                        or table.metadata.num_rows <= 0
                    ):
                        raise ValueError("Invalid evaluation parquet schema/rows")
                    checked[path] = table.metadata.num_rows
            expected = {
                item["name"]: item["num_examples"]
                for item in info.get(config["config_name"], {}).get("splits", [])
            }
            if (
                split["split"] in expected
                and sum(checked[path] for path in paths) != expected[split["split"]]
            ):
                raise ValueError(
                    "Evaluation parquet row count differs from pinned metadata"
                )
            layouts[config["config_name"]][split["split"]] = [
                str(path) for path in sorted(paths)
            ]
    if muse:
        required = {
            "knowmem": {"forget_qa", "forget_qa_icl", "retain_qa", "retain_qa_icl"},
            "privleak": {"forget", "holdout"},
            "verbmem": {"forget"},
        }
        if include_raw:
            required["raw"] = {"forget", "retain1"}
        if set(layouts) != set(required) or any(
            not splits.issubset(layouts[name]) for name, splits in required.items()
        ):
            raise ValueError("Incomplete pinned MUSE evaluation configurations/splits")
    return layouts


def validate_model_snapshot(path: Path) -> None:
    """Require a model config and a complete readable weight-shard set."""
    from safetensors import safe_open

    if not read_json(path / "config.json").get("model_type"):
        raise ValueError("Model snapshot has no model_type")
    mapping = None
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if (path / index).exists():
            mapping = read_json(path / index).get("weight_map")
            if not mapping:
                raise ValueError("Empty model shard index")
            files = set(mapping.values())
            break
    else:
        files = {
            name
            for name in ("model.safetensors", "pytorch_model.bin")
            if (path / name).is_file()
        }
    if not files:
        raise ValueError("Pinned model weights are missing")
    observed = set()
    for name in sorted(files):
        shard = path / name
        if Path(name).name != name or not shard.is_file() or not shard.stat().st_size:
            raise ValueError(f"Missing or unsafe model shard: {name}")
        try:
            if shard.suffix == ".safetensors":
                with safe_open(shard, framework="pt") as handle:
                    keys = set(handle.keys())
            else:
                tensors = torch.load(
                    shard, map_location="cpu", weights_only=True, mmap=True
                )
                if not isinstance(tensors, dict) or any(
                    not isinstance(value, torch.Tensor) for value in tensors.values()
                ):
                    raise ValueError("Non-tensor model state")
                keys = set(tensors)
                del tensors
            if not keys or (
                mapping is not None and any(mapping.get(key) != name for key in keys)
            ):
                raise ValueError("Shard keys differ from index")
            observed.update(keys)
        except Exception as error:
            raise ValueError(f"Invalid model shard {name}: {error}") from error
    if mapping is not None and observed != set(mapping):
        raise ValueError("Model shard index has missing weights")


class _WMDPRows:
    def __init__(self, rows, tokenizer, indices=None):
        self.rows, self.tokenizer, self.indices = rows, tokenizer, indices

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source = index if self.indices is None else self.indices[index]
        value = self.tokenizer(
            self.rows[source]["text"],
            max_length=1024,
            padding="max_length",
            truncation=True,
        )
        tokens = torch.tensor(value["input_ids"])
        return {
            "input_ids": tokens,
            "labels": tokens.clone(),
            "attention_mask": torch.tensor(value["attention_mask"]),
        }


def wmdp_pools(forget_rows, retain_rows, tokenizer):
    """Official WAGLE forget_ratio=1.0 still permutes with dataset seed 1000."""
    indices = random.Random(1000).sample(range(len(forget_rows)), len(forget_rows))
    return _WMDPRows(forget_rows, tokenizer, indices), _WMDPRows(retain_rows, tokenizer)


def muse_examples(texts, tokenizer, *, max_length: int = 2048) -> list[dict]:
    """Match PretrainingDataset's document token stream and decode/reencode."""
    separator = tokenizer("\n\n", add_special_tokens=False)["input_ids"]
    stream = []
    for start in range(0, len(texts), 32):
        tokenized = tokenizer(
            list(texts[start : start + 32]), add_special_tokens=False
        )["input_ids"]
        for offset, tokens in enumerate(tokenized):
            if start + offset:
                stream.extend(separator)
            stream.extend(tokens)
    prefix_length = len(tokenizer("", add_special_tokens=True)["input_ids"])
    masked_prefix = max(1, prefix_length)
    examples = []
    for start in range(0, len(stream), max_length):
        text = tokenizer.decode(stream[start : start + max_length])
        tokens = tokenizer(text, add_special_tokens=True)["input_ids"][
            : prefix_length + max_length
        ]
        labels = [-100] * masked_prefix + tokens[masked_prefix:]
        examples.append(
            {
                "input_ids": torch.tensor(tokens),
                "labels": torch.tensor(labels),
                "attention_mask": torch.ones(len(tokens), dtype=torch.long),
            }
        )
    return examples


def _snapshot_specs(spec: dict) -> dict:
    values = {
        "model": dict(spec["model"]),
        "tokenizer": dict(spec.get("tokenizer", spec["model"])),
    }
    if "dataset" in spec:
        values["dataset"] = {**spec["dataset"], "repo_type": "dataset"}
        values["eval_logs"] = dict(spec["evaluation"]["retain_logs"])
    else:
        values.update(
            {
                name: {**item, "repo_type": "dataset"}
                for name, item in spec["datasets"].items()
            }
        )
    values.update(
        {
            name: {**item, "repo_type": "dataset"}
            for name, item in spec["evaluation"].get("datasets", {}).items()
        }
    )
    return values


def _revisions(spec: dict) -> dict:
    values = {
        "model": spec["model"]["revision"],
        "tokenizer": spec.get("tokenizer", spec["model"])["revision"],
    }
    if "dataset" in spec:
        values["dataset"] = spec["dataset"]["revision"]
    else:
        values.update(
            {name: item["revision"] for name, item in spec["datasets"].items()}
        )
    return values


def prepare_streams(
    dataset: str, directory: Path, forget, retain, spec: dict, contract_digest: str
) -> dict:
    """Publish or validate both streams; paper content hashes are mandatory."""
    output = {}
    for stage in ("stage1", "stage2"):
        plan = (
            wmdp_plan(len(forget), len(retain))
            if dataset == "wmdpall"
            else muse_plan(len(forget), len(retain), stage)
        )
        expected = spec["traces"][f"{stage}_sha256"]
        if (
            stage == "stage2"
            and "stage2_effective_consumption_sha256" in spec["traces"]
        ):
            if (
                consumption_hash(plan)
                != spec["traces"]["stage2_effective_consumption_sha256"]
            ):
                raise ValueError("Stage 2 consumption hash differs from contract")
        destination = directory / stage
        if destination.exists() and any(destination.iterdir()):
            manifest = validate_stream(
                destination,
                expected_hash=expected,
                revisions=_revisions(spec),
                contract_digest=contract_digest,
            )
            if manifest["plan"] != plan:
                raise ValueError("Existing trace plan differs from source pools")
        else:
            payload = build_payload(
                forget, retain, plan, pad_token_id=None if dataset == "wmdpall" else 2
            )
            manifest = publish_stream(
                destination,
                payload,
                plan,
                dataset=dataset,
                stage=stage,
                revisions=_revisions(spec),
                contract_digest=contract_digest,
                expected_hash=expected,
            )
            del payload
        output[stage] = str((destination / "pairs.pt").absolute())
        output[f"{stage}_sha256"] = manifest["pair_stream_sha256"]
        if stage == "stage1":
            output["score_steps"] = manifest["stage1_score_steps"]
            output["forget_score_batches"] = manifest["stage1_wagle_forget_batches"]
            output["retain_score_batches"] = manifest["stage1_wagle_retain_batches"]
    return output


def _load_pools(dataset: str, snapshots: dict, work: Path):
    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshots["tokenizer"]["path"],
        local_files_only=True,
        **({"use_fast": False} if dataset == "wmdpall" else {}),
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("Pinned tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if dataset == "wmdpall":

        def parquet(label, subdir):
            files = sorted((Path(snapshots[label]["path"]) / subdir).glob("*.parquet"))
            if not files:
                raise FileNotFoundError(
                    f"Missing pinned parquet files: {label}/{subdir}"
                )
            return load_dataset(
                "parquet",
                data_files=[str(path) for path in files],
                split="train",
                cache_dir=str(work / "dataset_cache"),
            )

        forget = concatenate_datasets(
            [
                parquet("wmdp_corpora", "cyber-forget-corpus"),
                parquet("wmdp_bio_forget", "data"),
            ]
        )
        retain = concatenate_datasets(
            [
                parquet("wmdp_corpora", "cyber-retain-corpus"),
                parquet("wmdp_corpora", "bio-retain-corpus"),
            ]
        )
        if len(forget) != 25453:
            raise ValueError(f"Pinned WMDPALL forget count changed: {len(forget)}")
        return wmdp_pools(forget, retain, tokenizer)
    if tokenizer.pad_token_id != 2:
        raise ValueError("Pinned MUSE tokenizer padding must be EOS id 2")
    pools = []
    spec = load_contract()["datasets"][dataset]["dataset"]
    raw = validate_evaluation_snapshot(
        snapshots["dataset"], {**spec, "repo_type": "dataset"}, include_raw=True
    )["raw"]
    for split in ("forget", "retain1"):
        # HF snapshot wildcard inference drops blob symlinks in this datasets
        # version. Explicit, validated raw shards preserve source row order.
        from datasets import DownloadConfig

        rows = load_dataset(
            "parquet",
            name="raw",
            data_files={split: raw[split]},
            split=split,
            cache_dir=str(work / "dataset_cache"),
            download_config=DownloadConfig(local_files_only=True),
        )
        pools.append(muse_examples(rows["text"], tokenizer))
    expected = (407, 803) if dataset == "muse-news" else (553, 105)
    if tuple(map(len, pools)) != expected:
        raise ValueError(
            f"Pinned MUSE chunk counts changed: {tuple(map(len, pools))} != {expected}"
        )
    return tuple(pools)


def _validate_logs(record: dict, spec: dict) -> None:
    if "retain_logs" not in spec["evaluation"]:
        return
    path = Path(record["retain_logs"])
    if (
        not path.is_file()
        or sha256_file(path) != spec["evaluation"]["retain_logs_sha256"]
    ):
        raise ValueError("Retain evaluation log content hash differs from contract")
    if not read_json(path):
        raise ValueError("Retain evaluation log is empty")


def _validate_execution_metadata(
    value: dict, manifests: dict, spec: dict, dataset: str, stage1_resources: dict
) -> None:
    stage1, stage2 = manifests["stage1"], manifests["stage2"]
    expected = {
        "score_steps": stage1["stage1_score_steps"],
        "forget_score_batches": stage1["stage1_wagle_forget_batches"],
        "retain_score_batches": stage1["stage1_wagle_retain_batches"],
        "stage1_sha256": stage1["pair_stream_sha256"],
        "stage2_sha256": stage2["pair_stream_sha256"],
    }
    for key, observed in expected.items():
        if type(value.get(key)) is not type(observed) or value[key] != observed:
            raise ValueError(
                f"Prepared execution metadata differs from validated trace: {key}"
            )
    score_steps = spec["traces"].get("tcus_score_steps")
    if score_steps is not None and score_steps != expected["score_steps"]:
        raise ValueError("Trace score steps differ from contract dimensions")
    accumulation = stage1_resources.get("gradient_accumulation_steps")
    if (
        accumulation is not None
        and stage1["stage2_gradient_accumulation_steps"] != accumulation
    ):
        raise ValueError("Stage 1 accumulation differs from contract dimensions")
    recipe = spec.get("stage2")
    if recipe is not None:
        count = recipe["microbatches"]
        available = stage2["stage2_microbatches"]
        consumed = (
            recipe["max_steps"]
            * recipe["gradient_accumulation_steps"]
            * recipe["per_device_train_batch_size"]
        )
        if (
            count != consumed
            or count <= 0
            or count > available
            or stage2["stage2_gradient_accumulation_steps"]
            != recipe["gradient_accumulation_steps"]
        ):
            raise ValueError("Stage 2 trace differs from contract dimensions")


def validate_assets(work: Path, datasets: tuple[str, ...], *, contract=None) -> dict:
    selected = load_contract() if contract is None else contract
    record = read_assets(Path(work) / "assets.json", contract_hash(selected))
    for dataset in datasets:
        if dataset not in record["datasets"]:
            raise ValueError(f"Missing prepared dataset: {dataset}")
        value = record["datasets"][dataset]
        spec = selected["datasets"][dataset]
        for name, pinned in _snapshot_specs(spec).items():
            validator = (
                validate_evaluation_snapshot
                if name in spec["evaluation"].get("datasets", {})
                or (dataset.startswith("muse-") and name == "dataset")
                else validate_snapshot
            )
            if dataset.startswith("muse-") and name == "dataset":
                validator(value["snapshots"][name], pinned, include_raw=True)
            else:
                validator(value["snapshots"][name], pinned)
        for name in ("model", "tokenizer"):
            if value[name] != value["snapshots"][name]["path"]:
                raise ValueError(
                    f"Prepared {name} path differs from validated snapshot"
                )
        validate_model_snapshot(Path(value["model"]))
        manifests = {}
        for stage in ("stage1", "stage2"):
            declared = Path(value[stage])
            validated = declared.parent / "pairs.pt"
            if (
                not declared.is_absolute()
                or declared.name != "pairs.pt"
                or not declared.is_file()
                or declared.resolve(strict=True) != validated.resolve(strict=True)
            ):
                raise ValueError(
                    f"Prepared trace path is not the validated pairs.pt file: {stage}"
                )
            manifest = validate_stream(
                validated.parent,
                expected_hash=spec["traces"][f"{stage}_sha256"],
                revisions=_revisions(spec),
                contract_digest=contract_hash(selected),
            )
            expected_dataset = "WMDPALL" if dataset == "wmdpall" else dataset
            if manifest["stage"] != stage or manifest["dataset"] != expected_dataset:
                raise ValueError("Trace dataset/stage mismatch")
            manifests[stage] = manifest
            expected_consumption = spec["traces"].get(
                f"{stage}_effective_consumption_sha256"
            )
            if (
                expected_consumption
                and manifest["effective_consumption_trace_sha256"]
                != expected_consumption
            ):
                raise ValueError("Trace consumption hash differs from contract")
        _validate_execution_metadata(
            value, manifests, spec, dataset, selected.get("stage1_resources", {})
        )
        _validate_logs(value, spec)
    return record


def prepare_assets(
    repo: Path,
    work: Path,
    datasets: tuple[str, ...],
    *,
    cache_dir: Path | None = None,
    local_files_only: bool = True,
) -> dict:
    """Prepare inputs automatically; offline by default for foreground preflight.

    Network-enabled prefetch is opt-in via the Python API. The public runners
    consume cached snapshots; deployment-specific download environments belong
    to the invoking process, never to these portable modules.
    """
    contract = load_contract()
    if not datasets or not set(datasets).issubset(contract["datasets"]):
        raise ValueError("Unknown or empty dataset selection")
    work = Path(work).absolute()
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".assets.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if (work / "assets.json").exists():
            record = read_assets(work / "assets.json", contract_hash(contract))
            validate_assets(work, tuple(record["datasets"]), contract=contract)
        else:
            record = {
                "schema_version": 1,
                "contract_hash": contract_hash(contract),
                "datasets": {},
            }
        for dataset in datasets:
            if dataset in record["datasets"]:
                continue
            spec = contract["datasets"][dataset]
            snapshots = {
                name: resolve_snapshot(
                    pinned, cache_dir, local_files_only=local_files_only
                )
                for name, pinned in _snapshot_specs(spec).items()
            }
            value = {
                "snapshots": snapshots,
                "model": snapshots["model"]["path"],
                "tokenizer": snapshots["tokenizer"]["path"],
            }
            for name, pinned in spec["evaluation"].get("datasets", {}).items():
                validate_evaluation_snapshot(
                    snapshots[name], {**pinned, "repo_type": "dataset"}
                )
            if dataset.startswith("muse-"):
                validate_evaluation_snapshot(
                    snapshots["dataset"],
                    {**spec["dataset"], "repo_type": "dataset"},
                    include_raw=True,
                )
            validate_model_snapshot(Path(value["model"]))
            if "eval_logs" in snapshots:
                value["retain_logs"] = str(
                    Path(snapshots["eval_logs"]["path"])
                    / spec["evaluation"]["retain_logs"]["path"]
                )
                _validate_logs(value, spec)
            forget, retain = _load_pools(dataset, snapshots, work / dataset)
            value.update(
                prepare_streams(
                    dataset,
                    work / dataset / "data",
                    forget,
                    retain,
                    spec,
                    contract_hash(contract),
                )
            )
            del forget, retain
            record["datasets"][dataset] = value
            publish_assets(work / "assets.json", record)
        return validate_assets(work, datasets, contract=contract)
