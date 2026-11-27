"""Lossless exclusive per-model training evidence with two-level content seals."""

import gzip
import hashlib
import json
import zlib

SCHEMA = "n1-lossless-training-evidence-shard-v1"
FIELDS = ("losses", "measurements", "rows")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def validate(record, identity, expected):
    if (
        not {"seed", "arm", "protocol", "phase"} <= set(identity)
        or identity.get("seed") not in (11, 12, 13)
        or identity.get("arm") not in ("position", "terminal", "cut")
        or identity.get("phase") not in ("complete", "partial")
        or not isinstance(identity.get("protocol"), dict)
        or type(record.get("completed_updates")) is not int
    ):
        raise ValueError("explicit paired model/protocol/phase and completed count required")
    if set(expected) != set(FIELDS) or any(type(v) is not int or v < 0 for v in expected.values()):
        raise ValueError("all three exact nonnegative evidence counts required")
    if any(not isinstance(record.get(k), list) or len(record[k]) != expected[k] for k in FIELDS):
        raise ValueError("full loss/measurement/row counts differ")
    if (
        expected["losses"] != expected["measurements"]
        or record.get("completed_updates") != expected["losses"]
    ):
        raise ValueError("one loss and measurement per completed update required")
    if any(record.get(k) != identity.get(k) for k in ("seed", "arm", "protocol", "phase")):
        raise ValueError("shard model/protocol/phase identity differs")


def write_new(path, record, identity, expected):
    validate(record, identity, expected)
    raw = (
        json.dumps(
            dict(schema=SCHEMA, identity=identity, counts=expected, record=record),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    with path.open("xb") as stream:
        stream.write(compressed)
    return dict(
        path=str(path),
        schema=SCHEMA,
        identity=identity,
        counts=expected,
        compressed_sha256=digest(compressed),
        compressed_bytes=len(compressed),
        uncompressed_sha256=digest(raw),
        uncompressed_bytes=len(raw),
    )


def read_checked(path, seal, *, identity=None):
    compressed = path.read_bytes()
    if (
        len(compressed) != seal["compressed_bytes"]
        or digest(compressed) != seal["compressed_sha256"]
    ):
        raise ValueError("compressed evidence seal differs")
    try:
        raw = gzip.decompress(compressed)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError("invalid compressed evidence") from error
    if len(raw) != seal["uncompressed_bytes"] or digest(raw) != seal["uncompressed_sha256"]:
        raise ValueError("uncompressed evidence seal differs")
    payload = json.loads(raw)
    if (
        payload.get("schema") != SCHEMA
        or seal.get("schema") != SCHEMA
        or payload.get("identity") != seal["identity"]
        or payload.get("counts") != seal["counts"]
        or (identity is not None and identity != payload["identity"])
    ):
        raise ValueError("evidence schema, identity or count seal differs")
    validate(payload["record"], payload["identity"], payload["counts"])
    return payload["record"]
