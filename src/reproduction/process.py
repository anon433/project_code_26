"""Process evidence, interprocess GPU ownership, and between-job reclamation."""

from __future__ import annotations

import fcntl
import os
import re
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


def training_environment(env=None):
    clean = {
        key: value
        for key, value in (os.environ if env is None else env).items()
        if not key.lower().endswith("_proxy")
    }
    clean.update(
        CUDA_VISIBLE_DEVICES="0,1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        WORLD_SIZE="1",
    )
    for key in (
        "RANK",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "GROUP_RANK",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        clean.pop(key, None)
    clean.pop("PYTHONPATH", None)
    clean.pop("EXTRA_MASK_FRACS", None)
    return clean


def run_logged(command, cwd, log, *, env=None, on_start=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as handle:
        handle.write(shlex.join(command) + "\n")
        with subprocess.Popen(
            command,
            cwd=cwd,
            env=training_environment(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
        ) as child:
            if on_start:
                on_start(process_identity(child.pid))
            code = child.wait()
    if code:
        raise RuntimeError(f"Process failed ({code}); no valid result: {log}")
    return child


@contextmanager
def gpu_lock(path, *, on_wait=None, poll=5.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if on_wait:
                    on_wait()
                time.sleep(poll)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def screen_present(listing, name):
    return any(
        re.search(
            rf"^\s*\d+\.{re.escape(name)}\s+(?:\([^)]*\)\s+)?\((?:Detached|Attached)\)",
            line,
        )
        for line in listing.splitlines()
    )


def screen_listing(screen="screen"):
    return subprocess.run([screen, "-ls"], text=True, capture_output=True).stdout


def process_identity(pid):
    root = Path("/proc") / str(pid)
    deadline = time.monotonic() + 1
    while True:
        # Fields after the final ')' start at process state (field 3).
        stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
        command = (root / "cmdline").read_bytes().decode().rstrip("\0")
        if stat[0] == "Z":
            raise ProcessLookupError("Process is a zombie")
        if command:
            return {"pid": pid, "start_ticks": stat[19], "command": command.split("\0")}
        if time.monotonic() >= deadline:
            raise ProcessLookupError("Process command is unavailable")
        time.sleep(0.01)


def process_alive(identity):
    try:
        return process_identity(identity["pid"]) == {
            key: identity[key] for key in ("pid", "start_ticks", "command")
        }
    except (OSError, KeyError):
        return False


def validate_launch(identity, name, listing):
    if identity.get("session") != name or not screen_present(listing, name):
        raise ValueError("Expected exact live screen session is missing")
    if not process_alive(identity):
        raise ValueError("Runner process identity mismatch or stale PID")
    log = Path(identity["log"])
    if not log.is_file() or not log.stat().st_size:
        raise ValueError("Runner log missing or empty")


def memory_snapshot(cgroup=Path("/sys/fs/cgroup")):
    maximum = (cgroup / "memory.max").read_text().strip()
    if maximum == "max":
        raise RuntimeError("A finite cgroup memory limit is required")
    current = int((cgroup / "memory.current").read_text())
    stats = dict(
        line.split() for line in (cgroup / "memory.stat").read_text().splitlines()
    )
    return {
        "limit": int(maximum),
        "current": current,
        "headroom": int(maximum) - current,
        **{key: int(stats[key]) for key in ("file", "active_file", "inactive_file")},
    }


def require_headroom(snapshot, minimum):
    if snapshot["headroom"] < minimum:
        raise RuntimeError(
            f"Insufficient cgroup headroom: {snapshot['headroom']} < {minimum}"
        )


def reclaim_cache(work, *, child=None, drop_caches=Path("/proc/sys/vm/drop_caches")):
    if child is not None and child.poll() is None:
        raise RuntimeError("Cannot reclaim cache during a live process")
    os.sync()
    try:
        drop_caches.write_text("3\n")
    except OSError:
        for file in work.rglob("*"):
            if (
                file.is_file()
                and not file.is_symlink()
                and file.stat().st_uid == os.getuid()
            ):
                try:
                    with file.open("rb") as handle:
                        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                except OSError:
                    continue
