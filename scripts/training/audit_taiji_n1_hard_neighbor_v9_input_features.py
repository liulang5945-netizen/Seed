"""Stratify consumed v9 confirmation outcomes by input features, without inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "plans/manifests/n1_hard_neighbor_v9_20261017.json"
CONFIRMATION = ROOT / "reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json"
PLAN = ROOT / "plans/reference/PLAN-N1-104_v9_confirmation_input_feature_audit_20261017.md"
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v9_input_feature_audit_20261017.json"

SEEDS = (11, 12, 13)
ARMS = ("position", "cut", "terminal")
KINDS = ("lookup", "revision", "transition", "unknown")
PRIOR_FAMILY = "prior_family_descriptive"
NEW_FAMILY = "hard_neighbor_v9"
CPU_LIMIT_SECONDS = 30
MANIFEST_SHA256 = "d93dd6c6e0d98125ef1c91ed7ad8aec07a46f6621fb597dbffa6e889f29ac60f"
CONFIRMATION_SHA256 = "7b9b7e7cccf27a5e9ba6f9e002b1b4df09347fb7264b3827f9eb6cdab44a1802"
PLAN_SHA256 = "5943bd3f0f167ca8287a14dd4c7ff56196fda2b70e8d4bc3157c7f9045bacc0a"

# A setup-time preview decoded the v9 sacrifice payloads before this bounded
# reader was implemented. Keep that breach visible in every generated receipt.
PRE_EXECUTION_SCOPE_DEVIATION = {
    "occurred": True,
    "affected_manifest_items": 20,
    "boundary": "v9 sacrifice item payload fields were decoded during setup inspection",
    "used_in_analysis": False,
    "reader_decodes_sacrifice_payloads": False,
    "v7_v8_or_v3_v5_v6_sources_read": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JsonStream:
    """Scan JSON values while decoding only explicitly selected objects."""

    def __init__(self, path: Path):
        self._stream = Path(path).open("r", encoding="utf-8")
        self._lookahead: str | None = None
        self._lookahead_cookie = None

    def close(self) -> None:
        self._stream.close()

    def mark(self):
        return self._lookahead_cookie if self._lookahead is not None else self._stream.tell()

    def seek(self, cookie) -> None:
        self._stream.seek(cookie)
        self._lookahead = None
        self._lookahead_cookie = None

    def peek(self) -> str:
        if self._lookahead is None:
            self._lookahead_cookie = self._stream.tell()
            self._lookahead = self._stream.read(1)
        return self._lookahead

    def get(self) -> str:
        if self._lookahead is None:
            return self._stream.read(1)
        value = self._lookahead
        self._lookahead = None
        self._lookahead_cookie = None
        return value

    def skip_ws(self) -> None:
        while self.peek() and self.peek() in " \t\r\n":
            self.get()

    def expect(self, expected: str) -> None:
        actual = self.get()
        if actual != expected:
            raise ValueError(f"expected {expected!r}, got {actual!r}")

    def read_string(self) -> str:
        raw = ['"']
        self.expect('"')
        escaped = False
        while True:
            char = self.get()
            if not char:
                raise ValueError("unterminated JSON string")
            raw.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return json.loads("".join(raw))

    def read_scalar(self) -> Any:
        self.skip_ws()
        if self.peek() == '"':
            return self.read_string()
        raw = []
        while self.peek() and self.peek() not in ",]}\r\n \t":
            raw.append(self.get())
        if not raw:
            raise ValueError("expected scalar JSON value")
        value = json.loads("".join(raw))
        if isinstance(value, (dict, list)):
            raise ValueError("expected scalar JSON value")
        return value

    def skip_value(self) -> None:
        self.skip_ws()
        first = self.get()
        if not first:
            raise ValueError("unexpected end of JSON while skipping a value")
        if first == '"':
            escaped = False
            while True:
                char = self.get()
                if not char:
                    raise ValueError("unterminated string while skipping a value")
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    return
        if first in "[{":
            curly = 1 if first == "{" else 0
            square = 1 if first == "[" else 0
            in_string = False
            escaped = False
            while curly or square:
                char = self.get()
                if not char:
                    raise ValueError("unterminated JSON value")
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                elif char == '"':
                    in_string = True
                elif char == "{":
                    curly += 1
                elif char == "}":
                    curly -= 1
                elif char == "[":
                    square += 1
                elif char == "]":
                    square -= 1
                if curly < 0 or square < 0:
                    raise ValueError("unbalanced JSON value")
            return
        while self.peek() and self.peek() not in ",]}":
            self.get()

    def read_object_at(self, cookie):
        cursor = self.mark()
        self.seek(cookie)
        capture = []
        curly = 1
        square = 0
        in_string = False
        escaped = False
        self.expect("{")
        capture.append("{")
        while curly:
            char = self.get()
            if not char:
                raise ValueError("unterminated selected JSON object")
            capture.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char == "{":
                curly += 1
            elif char == "}":
                curly -= 1
            elif char == "[":
                square += 1
            elif char == "]":
                square -= 1
        if square:
            raise ValueError("unbalanced selected JSON object")
        try:
            return json.loads("".join(capture))
        finally:
            self.seek(cursor)


def _read_manifest_episode(stream: JsonStream) -> tuple[dict[str, Any], dict | None]:
    start = stream.mark()
    stream.expect("{")
    metadata: dict[str, Any] = {}
    while True:
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        key = stream.read_string()
        if key in metadata:
            raise ValueError(f"duplicate manifest item field {key!r}")
        stream.skip_ws()
        stream.expect(":")
        stream.skip_ws()
        if key in {"id", "split", "subset"}:
            metadata[key] = stream.read_scalar()
            if not isinstance(metadata[key], str):
                raise ValueError(f"manifest {key} must be a string")
        else:
            stream.skip_value()
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        stream.expect(",")

    if set(metadata) != {"id", "split", "subset"}:
        raise ValueError("manifest item must expose exactly id/split/subset metadata")
    if metadata["split"] != "holdout":
        raise ValueError("v9 confirmation manifest item is outside the expected holdout split")
    if metadata["subset"] == "sacrifice":
        return metadata, None
    if metadata["subset"] != "confirm":
        raise ValueError("unexpected v9 manifest subset")

    item = stream.read_object_at(start)
    if any(item.get(key) != value for key, value in metadata.items()):
        raise ValueError("decoded confirmation item differs from its streamed metadata")
    return metadata, item


def stream_manifest(path: Path = MANIFEST) -> tuple[dict[str, dict], int]:
    stream = JsonStream(path)
    confirm_items: dict[str, dict] = {}
    sacrifice_count = 0
    item_count = 0
    found_episodes = False
    try:
        stream.skip_ws()
        stream.expect("{")
        while True:
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            key = stream.read_string()
            stream.skip_ws()
            stream.expect(":")
            stream.skip_ws()
            if key == "episodes":
                if found_episodes:
                    raise ValueError("duplicate manifest episodes field")
                found_episodes = True
                stream.expect("[")
                stream.skip_ws()
                if stream.peek() != "]":
                    while True:
                        metadata, item = _read_manifest_episode(stream)
                        item_count += 1
                        if item is None:
                            sacrifice_count += 1
                        else:
                            item_id = metadata["id"]
                            if item_id in confirm_items:
                                raise ValueError("duplicate v9 confirmation item ID")
                            confirm_items[item_id] = item
                        stream.skip_ws()
                        if stream.peek() == "]":
                            break
                        stream.expect(",")
                        stream.skip_ws()
                stream.expect("]")
            else:
                stream.skip_value()
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            stream.expect(",")
        stream.skip_ws()
        if stream.peek():
            raise ValueError("trailing data after v9 manifest")
    finally:
        stream.close()

    if not found_episodes or item_count != 40 or sacrifice_count != 20 or len(confirm_items) != 20:
        raise ValueError("v9 manifest must contain 20 confirm and 20 sacrifice items")
    return confirm_items, sacrifice_count


def _read_outcome_row(stream: JsonStream) -> tuple[str, dict | None]:
    start = stream.mark()
    stream.expect("{")
    family = None
    while True:
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        key = stream.read_string()
        stream.skip_ws()
        stream.expect(":")
        stream.skip_ws()
        if key == "family":
            if family is not None:
                raise ValueError("confirmation row has duplicate family field")
            family = stream.read_scalar()
        else:
            stream.skip_value()
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        stream.expect(",")
    if not isinstance(family, str):
        raise ValueError("confirmation row lacks string family field")
    if family != NEW_FAMILY:
        return family, None

    cursor = stream.mark()
    stream.seek(start)
    selected: dict[str, Any] = {}
    try:
        stream.expect("{")
        while True:
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            key = stream.read_string()
            if key in selected:
                raise ValueError(f"duplicate outcome row field {key!r}")
            stream.skip_ws()
            stream.expect(":")
            stream.skip_ws()
            if key in {"family", "id", "kind", "seed", "arm", "free_exact"}:
                selected[key] = stream.read_scalar()
            else:
                stream.skip_value()
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            stream.expect(",")
    finally:
        stream.seek(cursor)
    if set(selected) != {"family", "id", "kind", "seed", "arm", "free_exact"}:
        raise ValueError("v9 outcome row lacks required feature/outcome fields")
    return family, selected


METADATA_FIELDS = {
    "schema",
    "status",
    "complete",
    "passed",
    "confirm_items",
    "model_queries",
    "v9_confirm_model_queries",
    "v9_new_family_model_queries",
    "prior_family_descriptive_model_queries",
    "parameter_updates",
    "optimizer_steps",
    "sources_unchanged",
    "can_promote",
    "TRAIN_SHA",
}


def stream_confirmation(path: Path = CONFIRMATION) -> tuple[dict, list[dict], dict[str, int]]:
    stream = JsonStream(path)
    metadata: dict[str, Any] = {}
    new_rows: list[dict] = []
    skipped_families: Counter[str] = Counter()
    found_rows = False
    try:
        stream.skip_ws()
        stream.expect("{")
        while True:
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            key = stream.read_string()
            stream.skip_ws()
            stream.expect(":")
            stream.skip_ws()
            if key == "rows":
                if found_rows:
                    raise ValueError("duplicate confirmation rows field")
                found_rows = True
                stream.expect("[")
                stream.skip_ws()
                if stream.peek() != "]":
                    while True:
                        family, row = _read_outcome_row(stream)
                        if row is None:
                            skipped_families[family] += 1
                        else:
                            new_rows.append(row)
                        stream.skip_ws()
                        if stream.peek() == "]":
                            break
                        stream.expect(",")
                        stream.skip_ws()
                stream.expect("]")
            elif key in METADATA_FIELDS:
                if key in metadata:
                    raise ValueError(f"duplicate confirmation metadata field {key}")
                metadata[key] = stream.read_scalar()
            else:
                stream.skip_value()
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            stream.expect(",")
        stream.skip_ws()
        if stream.peek():
            raise ValueError("trailing data after confirmation report")
    finally:
        stream.close()
    if not found_rows:
        raise ValueError("confirmation report has no rows array")
    return metadata, new_rows, dict(skipped_families)


def validate_confirmation_metadata(metadata: dict, skipped: dict[str, int]) -> None:
    expected = {
        "schema": "n1-hard-neighbor-v9-confirmation-result-v1",
        "status": "complete",
        "complete": True,
        "passed": False,
        "confirm_items": 20,
        "model_queries": 360,
        "v9_confirm_model_queries": 360,
        "v9_new_family_model_queries": 180,
        "prior_family_descriptive_model_queries": 180,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "sources_unchanged": True,
        "can_promote": False,
        "TRAIN_SHA": None,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("confirmation metadata differs from the frozen v9 receipt")
    if skipped != {PRIOR_FAMILY: 180}:
        raise ValueError("exactly 180 prior-family rows must be skipped without decoding")


def extract_input_features(item: dict) -> dict:
    item_id = item.get("id")
    kind = item.get("kind")
    entities = item.get("entities")
    observations = item.get("observations")
    if not isinstance(item_id, str) or kind not in KINDS:
        raise ValueError("confirmation item needs a known ID and kind")
    if not isinstance(entities, list) or not entities or not isinstance(entities[0], str):
        raise ValueError("confirmation item must identify its target entity")
    if not isinstance(observations, list) or any(not isinstance(row, str) for row in observations):
        raise ValueError("confirmation item observations must be strings")

    target = entities[0]
    positions = [
        index
        for index, observation in enumerate(observations, start=1)
        if observation.startswith(target + " ")
    ]
    expected_target_rows = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}[kind]
    if len(positions) != expected_target_rows:
        raise ValueError(f"unexpected target observation count for {kind}: {len(positions)}")

    answer_length = None
    if kind in {"lookup", "revision"}:
        answer = item.get("answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError("lookup/revision item must have a nonempty material answer")
        answer_length = len(answer)
    return {
        "id": item_id,
        "kind": kind,
        "target_row_position_1based": positions[0] if positions else None,
        "target_observation_positions_1based": positions,
        "material_answer_length_chars": answer_length,
    }


def group_outcomes(items: dict[str, dict], rows: list[dict]) -> dict:
    expected_ids = set(items)
    cells: dict[tuple[str, int | None, int | None], dict] = {}
    seen: set[tuple[int, str, str]] = set()
    endpoint_ids: dict[tuple[int, str], set[str]] = {
        (seed, arm): set() for seed in SEEDS for arm in ARMS
    }

    for row in rows:
        if row.get("family") != NEW_FAMILY:
            raise ValueError("analysis input must contain only v9 new-family rows")
        seed, arm = row.get("seed"), row.get("arm")
        item_id = row.get("id")
        if seed not in SEEDS or arm not in ARMS or item_id not in items:
            raise ValueError("unexpected seed, arm, or confirmation item ID")
        identity = (seed, arm, item_id)
        if identity in seen:
            raise ValueError("duplicate seed/arm/item outcome row")
        seen.add(identity)
        endpoint_ids[(seed, arm)].add(item_id)
        if row.get("kind") != items[item_id]["kind"]:
            raise ValueError("input kind and v9 outcome kind differ")
        if type(row.get("free_exact")) is not bool:
            raise ValueError("v9 outcome row free_exact must be boolean")

        features = items[item_id]
        key = (
            features["kind"],
            features["target_row_position_1based"],
            features["material_answer_length_chars"],
        )
        cell = cells.setdefault(key, {"item_ids": set(), "outcomes": defaultdict(list)})
        cell["item_ids"].add(item_id)
        cell["outcomes"][(seed, arm)].append(row["free_exact"])

    if len(rows) != 180 or any(ids != expected_ids for ids in endpoint_ids.values()):
        raise ValueError("each of nine endpoints must contain the same 20 v9 confirmation items")
    if any(len(cell["outcomes"].get((seed, arm), [])) != len(cell["item_ids"])
           for cell in cells.values() for seed in SEEDS for arm in ARMS):
        raise ValueError("feature cell has incomplete seed/arm outcome coverage")

    grouped = []
    for (kind, position, answer_length), cell in sorted(
        cells.items(),
        key=lambda entry: (
            KINDS.index(entry[0][0]),
            entry[0][1] is None,
            entry[0][1] or 0,
            entry[0][2] is None,
            entry[0][2] or 0,
        ),
    ):
        per_seed_arm = {}
        for seed in SEEDS:
            per_seed_arm[str(seed)] = {}
            for arm in ARMS:
                outcomes = cell["outcomes"][(seed, arm)]
                per_seed_arm[str(seed)][arm] = {
                    "denominator": len(outcomes),
                    "free_exact_correct": sum(outcomes),
                    "free_exact_incorrect": len(outcomes) - sum(outcomes),
                }
        grouped.append(
            {
                "kind": kind,
                "target_row_position_1based": position,
                "material_answer_length_chars": answer_length,
                "unique_confirmation_items": len(cell["item_ids"]),
                "per_seed_arm": per_seed_arm,
            }
        )

    return {
        "confirmation_items": len(items),
        "confirmation_outcome_rows": len(rows),
        "seed_count": len(SEEDS),
        "arms": list(ARMS),
        "feature_cells": grouped,
        "position_definition": (
            "1-based index in observations of the first row beginning with the target entity; "
            "all target-row positions are retained as an audit field; unknown has no target row"
        ),
        "answer_length_definition": "Python Unicode code-point count; reported for lookup/revision only",
    }


def run_audit() -> dict:
    if REPORT.exists():
        raise FileExistsError("PLAN-N1-104 report is write-once")
    started = time.process_time()
    source_hashes = {
        "v9_manifest_sha256": sha256(MANIFEST),
        "v9_confirmation_report_sha256": sha256(CONFIRMATION),
        "plan_sha256": sha256(PLAN),
    }
    if source_hashes != {
        "v9_manifest_sha256": MANIFEST_SHA256,
        "v9_confirmation_report_sha256": CONFIRMATION_SHA256,
        "plan_sha256": PLAN_SHA256,
    }:
        raise ValueError("PLAN-N1-104 source bytes differ from the frozen protocol")

    inputs, sacrifice_count = stream_manifest(MANIFEST)
    input_features = {item_id: extract_input_features(item) for item_id, item in inputs.items()}
    if set(input_features) != set(inputs) or sacrifice_count != 20:
        raise ValueError("manifest input feature extraction is incomplete")
    metadata, rows, skipped = stream_confirmation(CONFIRMATION)
    validate_confirmation_metadata(metadata, skipped)
    grouped = group_outcomes(input_features, rows)
    cpu_seconds = time.process_time() - started
    complete = cpu_seconds <= CPU_LIMIT_SECONDS
    report = {
        "schema": "n1-hard-neighbor-v9-input-feature-audit-v1",
        "status": "complete_with_protocol_deviation" if complete else "stopped_cpu_limit",
        "complete": complete,
        "exploratory_only": True,
        "causal_claim_allowed": False,
        "gate_recomputed": False,
        **source_hashes,
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "manifest_items_total": 40,
        "confirm_input_objects_decoded_by_reader": len(inputs),
        "sacrifice_payload_objects_skipped_by_reader": sacrifice_count,
        "v9_outcome_rows_decoded": len(rows),
        "prior_family_rows_skipped_without_decoding": skipped[PRIOR_FAMILY],
        "protocol_deviation": PRE_EXECUTION_SCOPE_DEVIATION,
        "input_feature_outcomes": grouped,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "new_face_queries": 0,
        "face_writes": 0,
        "can_promote": False,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(cpu_seconds, 6),
        "interpretation_limit": (
            "descriptive stratification of already-consumed v9 confirmation outcomes only; "
            "the preparation-time sacrifice-payload exposure is disclosed above; no causal, "
            "capability, C3, C4, retention, three-organ, or promotion claim"
        ),
    }
    write_new(REPORT, report)
    return report


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    parser.parse_args()
    report = run_audit()
    print(
        json.dumps(
            {
                "status": report["status"],
                "complete": report["complete"],
                "report": str(REPORT),
                "report_sha256": sha256(REPORT),
                "rows_analyzed": report["v9_outcome_rows_decoded"],
                "feature_cells": len(report["input_feature_outcomes"]["feature_cells"]),
                "cpu_seconds": report["cpu_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
