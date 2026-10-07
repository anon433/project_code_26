import hashlib
import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


logger = logging.getLogger("data")

_PERSISTENT_CACHE_SCHEMA_VERSION = 1
_PERSISTENT_CACHE_DIRNAME = "pretraining_token_stream_v1"
_TOKEN_STREAM_FILENAME = "token_stream.int32"
_METADATA_FILENAME = "metadata.json"


def _jsonable(value):
    """Convert Hydra values and tokenizer metadata into stable JSON values."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _canonical_json(value):
    """Return a stable JSON representation for cache identity material."""
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _fingerprint_payload(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@contextmanager
def _exclusive_file_lock(path):
    """Serialize cache publication without adding a third-party dependency."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def _tokenizer_parallelism_for_cache_build():
    """Use Rust tokenizer parallelism only while no data-loader workers exist."""
    variable = "TOKENIZERS_PARALLELISM"
    previous = os.environ.get(variable)
    os.environ[variable] = "true"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item


class PretrainingDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        text_key="text",
        max_length=2048,
        persistent_cache=False,
        persistent_cache_dir=None,
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunks = None
        self._token_stream = None
        self._token_count = 0
        if persistent_cache:
            self._initialize_persistent_cache(
                hf_args=hf_args,
                text_key=text_key,
                persistent_cache_dir=persistent_cache_dir,
            )
        else:
            self.chunks = self._chunk_raw_text(load_hf_dataset(**hf_args)[text_key])

    def _cache_identity(self, hf_args, text_key):
        """Inputs whose change must never reuse a token stream."""
        vocabulary = self.tokenizer.get_vocab()
        return {
            "schema_version": _PERSISTENT_CACHE_SCHEMA_VERSION,
            "hf_args": _jsonable(hf_args),
            "text_key": text_key,
            "max_length": self.max_length,
            "tokenizer": {
                "vocabulary": vocabulary,
                "added_vocabulary": getattr(
                    self.tokenizer, "get_added_vocab", lambda: {}
                )(),
                "special_tokens_map": getattr(self.tokenizer, "special_tokens_map", {}),
                "all_special_ids": list(getattr(self.tokenizer, "all_special_ids", [])),
            },
        }

    @staticmethod
    def _cache_root(persistent_cache_dir):
        if persistent_cache_dir is not None:
            return Path(persistent_cache_dir)
        datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        if datasets_cache is None:
            datasets_cache = str(Path.home() / ".cache" / "huggingface" / "datasets")
        return Path(datasets_cache) / _PERSISTENT_CACHE_DIRNAME

    @staticmethod
    def _read_json(path):
        try:
            with path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _write_json_atomic(path, payload):
        temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _valid_cache_metadata(self, cache_dir, metadata, identity, cache_key=None):
        if metadata is None:
            return False
        if metadata.get("schema_version") != _PERSISTENT_CACHE_SCHEMA_VERSION:
            return False
        if metadata.get("identity") != identity:
            return False
        if cache_key is not None and metadata.get("cache_key") != cache_key:
            return False
        fingerprint = metadata.get("dataset_fingerprint")
        token_count = metadata.get("token_count")
        if not isinstance(fingerprint, str) or not fingerprint:
            return False
        expected_cache_key = _fingerprint_payload(
            {"identity": identity, "dataset_fingerprint": fingerprint}
        )
        if metadata.get("cache_key") != expected_cache_key:
            return False
        if not isinstance(token_count, int) or token_count < 0:
            return False
        if metadata.get("token_dtype") != "int32":
            return False
        token_path = cache_dir / _TOKEN_STREAM_FILENAME
        expected_size = token_count * np.dtype(np.int32).itemsize
        try:
            return token_path.is_file() and token_path.stat().st_size == expected_size
        except OSError:
            return False

    def _open_token_stream(self, cache_dir, metadata):
        self._token_count = metadata["token_count"]
        if self._token_count == 0:
            self._token_stream = np.empty(0, dtype=np.int32)
            return
        self._token_stream = np.memmap(
            cache_dir / _TOKEN_STREAM_FILENAME,
            dtype=np.int32,
            mode="r",
            shape=(self._token_count,),
        )

    def _initialize_persistent_cache(self, hf_args, text_key, persistent_cache_dir):
        identity = self._cache_identity(hf_args, text_key)
        root = self._cache_root(persistent_cache_dir)
        root.mkdir(parents=True, exist_ok=True)
        lookup_key = _fingerprint_payload(identity)
        lookup_path = root / f"lookup-{lookup_key}.json"

        # A warm cache does not need to reopen the source Arrow/parquet dataset.
        lookup = self._read_json(lookup_path)
        if lookup is not None and lookup.get("identity") == identity:
            cache_key = lookup.get("cache_key")
            if isinstance(cache_key, str):
                cache_dir = root / cache_key
                metadata = self._read_json(cache_dir / _METADATA_FILENAME)
                if self._valid_cache_metadata(cache_dir, metadata, identity, cache_key):
                    self._open_token_stream(cache_dir, metadata)
                    logger.info(
                        "Pretraining token-stream cache hit: %s (%d tokens)",
                        cache_dir,
                        self._token_count,
                    )
                    return

        logger.info("Pretraining token-stream cache miss; building under %s", root)
        lock_path = root / f"lookup-{lookup_key}.lock"
        with _exclusive_file_lock(lock_path):
            # Another process may have completed the cache while this process waited.
            lookup = self._read_json(lookup_path)
            if lookup is not None and lookup.get("identity") == identity:
                cache_key = lookup.get("cache_key")
                if isinstance(cache_key, str):
                    cache_dir = root / cache_key
                    metadata = self._read_json(cache_dir / _METADATA_FILENAME)
                    if self._valid_cache_metadata(
                        cache_dir, metadata, identity, cache_key
                    ):
                        self._open_token_stream(cache_dir, metadata)
                        logger.info(
                            "Pretraining token-stream cache hit after wait: %s (%d tokens)",
                            cache_dir,
                            self._token_count,
                        )
                        return

            source = load_hf_dataset(**hf_args)
            dataset_fingerprint = getattr(source, "_fingerprint", None)
            if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
                raise ValueError(
                    "Persistent pretraining cache requires a Hugging Face dataset "
                    "with a non-empty Arrow _fingerprint."
                )
            cache_key = _fingerprint_payload(
                {"identity": identity, "dataset_fingerprint": dataset_fingerprint}
            )
            cache_dir = root / cache_key
            metadata = self._read_json(cache_dir / _METADATA_FILENAME)
            if self._valid_cache_metadata(cache_dir, metadata, identity, cache_key):
                self._write_json_atomic(
                    lookup_path, {"identity": identity, "cache_key": cache_key}
                )
                self._open_token_stream(cache_dir, metadata)
                logger.info(
                    "Pretraining token-stream cache hit: %s (%d tokens)",
                    cache_dir,
                    self._token_count,
                )
                return

            temporary_dir = root / (".build-" + uuid.uuid4().hex)
            temporary_dir.mkdir()
            try:
                token_count = self._write_token_stream(
                    source, text_key, temporary_dir / _TOKEN_STREAM_FILENAME
                )
                metadata = {
                    "schema_version": _PERSISTENT_CACHE_SCHEMA_VERSION,
                    "cache_key": cache_key,
                    "identity": identity,
                    "dataset_fingerprint": dataset_fingerprint,
                    "token_count": token_count,
                    "token_dtype": "int32",
                }
                self._write_json_atomic(temporary_dir / _METADATA_FILENAME, metadata)
                if cache_dir.exists():
                    # The directory failed validation above while holding this key's lock.
                    shutil.rmtree(cache_dir)
                os.replace(temporary_dir, cache_dir)
            except Exception:
                shutil.rmtree(temporary_dir, ignore_errors=True)
                raise

            self._write_json_atomic(
                lookup_path, {"identity": identity, "cache_key": cache_key}
            )
            self._open_token_stream(cache_dir, metadata)
            logger.info(
                "Pretraining token-stream cache built: %s (%d tokens)",
                cache_dir,
                self._token_count,
            )

    def _write_token_stream(self, source, text_key, output_path):
        """Build the newline-separated stream without retaining it in Python memory."""
        separator_ids = np.asarray(
            self.tokenizer("\n\n", add_special_tokens=False)["input_ids"],
            dtype=np.int32,
        )
        token_count = 0
        batch_size = 32
        with output_path.open("wb") as handle:
            with _tokenizer_parallelism_for_cache_build():
                for start in range(0, len(source), batch_size):
                    records = source[start : start + batch_size]
                    texts = list(records[text_key])
                    tokenized = self.tokenizer(texts, add_special_tokens=False)[
                        "input_ids"
                    ]
                    for offset, token_ids in enumerate(tokenized):
                        if start + offset:
                            separator_ids.tofile(handle)
                            token_count += len(separator_ids)
                        token_array = np.asarray(token_ids, dtype=np.int32)
                        token_array.tofile(handle)
                        token_count += len(token_array)
            handle.flush()
            os.fsync(handle.fileno())
        return token_count

    def _chunk_raw_text(self, raw_text):
        """Tokenize bounded document batches before forming fixed-length chunks.

        Passing an entire corpus as one string to the fast tokenizer becomes
        prohibitively slow for WMDP corpora.  Keeping document boundaries here
        preserves the same newline-separated stream while avoiding that
        pathological giant-input path.
        """
        separator_ids = self.tokenizer("\n\n", add_special_tokens=False)["input_ids"]
        full_token_sequence = []
        batch_size = 32
        for start in range(0, len(raw_text), batch_size):
            texts = list(raw_text[start : start + batch_size])
            tokenized = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
            for offset, token_ids in enumerate(tokenized):
                if start + offset:
                    full_token_sequence.extend(separator_ids)
                full_token_sequence.extend(token_ids)

        return [
            self.tokenizer.decode(full_token_sequence[start : start + self.max_length])
            for start in range(0, len(full_token_sequence), self.max_length)
        ]

    def __len__(self):
        if self.chunks is not None:
            return len(self.chunks)
        return (self._token_count + self.max_length - 1) // self.max_length

    def __getitem__(self, idx):
        if self.chunks is None:
            start = idx * self.max_length
            stop = min(start + self.max_length, self._token_count)
            text = self.tokenizer.decode(self._token_stream[start:stop].tolist())
        else:
            text = self.chunks[idx]
        item = preprocess_pretraining_instance(
            self.tokenizer, "", text, self.max_length
        )
        item["index"] = int(idx)
        return item
