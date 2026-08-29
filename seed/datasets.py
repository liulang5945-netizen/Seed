"""Native dataset contracts for Seed's raw-byte training path.

The native trainer accepts UTF-8 text documents and converts them to bytes at
the sensory boundary.  JSONL/JSON documents use the explicit ``text`` field;
plain UTF-8 text files are treated as one document.  No token vocabulary or
model-specific preprocessing is part of this contract.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NATIVE_TEXT_FIELD = "text"
_JSONL_SUFFIXES = {".jsonl", ".ndjson"}


class NativeDatasetError(ValueError):
    """Raised when a dataset cannot be consumed by the native byte trainer."""


@dataclass(frozen=True)
class NativeDatasetReport:
    """Machine-readable quality report for one native training dataset."""

    path: str
    format: str
    native_trainable: bool
    documents: int
    empty_documents: int
    invalid_records: int
    blank_lines: int
    total_text_bytes: int
    errors: tuple[str, ...]
    truncated: bool = False
    scanned_bytes: int = 0

    def estimated_total_text_bytes(self) -> int:
        """外推整个文件的文本字节数（未截断时即精确值）。"""
        if not self.truncated or self.scanned_bytes <= 0:
            return self.total_text_bytes
        try:
            file_size = Path(self.path).stat().st_size
        except OSError:
            return self.total_text_bytes
        if file_size <= self.scanned_bytes:
            return self.total_text_bytes
        ratio = file_size / self.scanned_bytes
        return int(self.total_text_bytes * ratio)

    def to_dict(self) -> dict[str, Any]:
        average = self.total_text_bytes / self.documents if self.documents else 0.0
        return {
            "path": self.path,
            "format": self.format,
            "native_trainable": self.native_trainable,
            "documents": self.documents,
            "empty_documents": self.empty_documents,
            "invalid_records": self.invalid_records,
            "blank_lines": self.blank_lines,
            "total_text_bytes": self.total_text_bytes,
            "average_text_bytes": round(average, 2),
            "truncated": self.truncated,
            "scanned_bytes": self.scanned_bytes,
            "estimated_total_text_bytes": self.estimated_total_text_bytes(),
            "errors": list(self.errors),
            "contract": {
                "encoding": "utf-8",
                "text_field": NATIVE_TEXT_FIELD,
                "representation": "raw-byte-stream",
            },
        }


def _text_from_record(record: Any, *, source: str, ordinal: int) -> str:
    if not isinstance(record, Mapping):
        raise NativeDatasetError(f"{source}:{ordinal} must be a JSON object")
    text = record.get(NATIVE_TEXT_FIELD)
    if not isinstance(text, str):
        raise NativeDatasetError(
            f"{source}:{ordinal} must contain a string field '{NATIVE_TEXT_FIELD}'"
        )
    return text


def _iter_jsonl_documents(path: Path) -> Iterator[str]:
    try:
        handle = path.open("r", encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NativeDatasetError(f"{path} is not valid UTF-8: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                yield _text_from_record(record, source=str(path), ordinal=line_number)
            except json.JSONDecodeError as exc:
                raise NativeDatasetError(
                    f"{path}:{line_number} is invalid JSON: {exc.msg}"
                ) from exc


def _iter_json_documents(path: Path) -> Iterator[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except UnicodeDecodeError as exc:
        raise NativeDatasetError(f"{path} is not valid UTF-8: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise NativeDatasetError(f"{path} is invalid JSON: {exc.msg}") from exc

    records = payload if isinstance(payload, list) else [payload]
    for ordinal, record in enumerate(records, start=1):
        yield _text_from_record(record, source=str(path), ordinal=ordinal)


def iter_native_documents(paths: Sequence[Path | str]) -> Iterator[str]:
    """Yield non-empty native documents in streaming order.

    JSONL is the canonical corpus format.  A JSON array of ``{"text": ...}``
    records and a plain UTF-8 text file are supported for small imports and
    previews.  Empty documents are ignored consistently by training and
    quality inspection.
    """

    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in _JSONL_SUFFIXES:
            documents = _iter_jsonl_documents(path)
        elif suffix == ".json":
            documents = _iter_json_documents(path)
        else:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise NativeDatasetError(f"{path} is not valid UTF-8: {exc}") from exc
            documents = iter((text,))
        for text in documents:
            if text:
                yield text


def inspect_native_dataset(
    path_value: Path | str, *, max_records: int | None = None
) -> NativeDatasetReport:
    """Inspect a dataset without importing any optional model runtime.

    ``max_records`` bounds how many JSONL lines are examined so that
    multi-gigabyte corpora can be validated in constant time.  When the scan
    stops early the report is flagged ``truncated``, ``scanned_bytes`` records
    the inspected prefix size, and ``estimated_total_text_bytes()`` extrapolates
    the full corpus size.  JSON and plain-text datasets are always read whole
    because their parsers materialise the entire file anyway.
    """

    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in _JSONL_SUFFIXES:
        dataset_format = "jsonl"
    elif suffix == ".json":
        dataset_format = "json"
    else:
        dataset_format = "utf-8-text"

    documents = 0
    empty_documents = 0
    invalid_records = 0
    blank_lines = 0
    total_text_bytes = 0
    truncated = False
    scanned_bytes = 0
    errors: list[str] = []

    def record_text(text: str) -> None:
        nonlocal documents, empty_documents, total_text_bytes
        if not text:
            empty_documents += 1
            return
        documents += 1
        total_text_bytes += len(text.encode("utf-8"))

    if suffix in _JSONL_SUFFIXES:
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if max_records is not None and line_number > max_records:
                        truncated = True
                        break
                    scanned_bytes += len(line.encode("utf-8"))
                    if not line.strip():
                        blank_lines += 1
                        continue
                    try:
                        record_text(
                            _text_from_record(
                                json.loads(line), source=str(path), ordinal=line_number
                            )
                        )
                    except (json.JSONDecodeError, NativeDatasetError) as exc:
                        invalid_records += 1
                        if len(errors) < 8:
                            errors.append(str(exc))
        except UnicodeDecodeError as exc:
            invalid_records += 1
            errors.append(f"{path} is not valid UTF-8: {exc}")
    elif suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            records = payload if isinstance(payload, list) else [payload]
            for ordinal, record in enumerate(records, start=1):
                try:
                    record_text(_text_from_record(record, source=str(path), ordinal=ordinal))
                except NativeDatasetError as exc:
                    invalid_records += 1
                    if len(errors) < 8:
                        errors.append(str(exc))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            invalid_records += 1
            errors.append(f"{path} cannot be parsed: {exc}")
    else:
        try:
            record_text(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            invalid_records += 1
            errors.append(f"{path} is not valid UTF-8: {exc}")

    native_trainable = documents > 0 and invalid_records == 0
    return NativeDatasetReport(
        path=str(path),
        format=dataset_format,
        native_trainable=native_trainable,
        documents=documents,
        empty_documents=empty_documents,
        invalid_records=invalid_records,
        blank_lines=blank_lines,
        total_text_bytes=total_text_bytes,
        errors=tuple(errors),
        truncated=truncated,
        scanned_bytes=scanned_bytes,
    )
