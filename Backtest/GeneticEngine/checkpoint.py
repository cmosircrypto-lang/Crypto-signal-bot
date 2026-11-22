"""Integrity-checked, atomic JSON checkpoints for resumable research jobs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base checkpoint error."""


class CheckpointIntegrityError(CheckpointError):
    """Checkpoint payload checksum does not match."""


class CheckpointCompatibilityError(CheckpointError):
    """Checkpoint belongs to different code, data or configuration."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def save_checkpoint(
    path: str | Path,
    *,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    """Atomically save a checkpoint and return its payload checksum.

    Metadata should include code/config hash, dataset hash, symbol, timeframe,
    last closed candle, candidate fingerprint and random seed where applicable.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": dict(metadata),
        "payload": dict(payload),
    }
    checksum = _sha256(body)
    envelope = {**body, "checksum": checksum}
    encoded = _canonical_json(envelope) + "\n"

    fd, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        # Best-effort directory fsync makes the rename durable on POSIX.
        try:
            directory_fd = os.open(str(destination.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return checksum


def load_checkpoint(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and verify a checkpoint; reject stale metadata instead of reusing it."""
    source = Path(path)
    try:
        envelope = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Cannot read checkpoint {source}: {exc}") from exc

    if envelope.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError("unsupported checkpoint schema version")
    checksum = envelope.get("checksum")
    body = {key: envelope[key] for key in ("schema_version", "metadata", "payload")}
    if not isinstance(checksum, str) or _sha256(body) != checksum:
        raise CheckpointIntegrityError("checkpoint checksum mismatch")

    actual_metadata = envelope.get("metadata", {})
    for key, expected in (expected_metadata or {}).items():
        if actual_metadata.get(key) != expected:
            raise CheckpointCompatibilityError(
                f"checkpoint metadata mismatch for {key!r}: "
                f"expected {expected!r}, found {actual_metadata.get(key)!r}"
            )
    return {
        "metadata": actual_metadata,
        "payload": envelope.get("payload", {}),
        "checksum": checksum,
    }
