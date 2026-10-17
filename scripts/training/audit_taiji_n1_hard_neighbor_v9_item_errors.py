"""Read-only item-level attribution of the consumed PLAN-N1-102 confirmation rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "plans/reference/PLAN-N1-103_v9_item_level_error_attribution_20261017.md"
CONFIRMATION = ROOT / "reports/taiji_n1_hard_neighbor_v9_confirmation_20261017.json"
REPORT = ROOT / "reports/taiji_n1_hard_neighbor_v9_item_error_attribution_20261017.json"
SEEDS = (11, 12, 13)
KINDS = ("lookup", "revision", "transition", "unknown")
ARMS = ("position", "cut", "terminal")
PRIOR_FAMILY = "prior_family_descriptive"
NEW_FAMILY = "hard_neighbor_v9"
CPU_LIMIT_SECONDS = 30
CONFIRMATION_SHA256 = "7b9b7e7cccf27a5e9ba6f9e002b1b4df09347fb7264b3827f9eb6cdab44a1802"
PLAN_SHA256 = "545f1fb4cb6934510d18107b46dd4094b8c59a73d26a929553db8db4fe9b7ecc"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JsonStream:
    """Scan JSON values without decoding skipped row payloads."""

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

    def read_string(self) -> tuple[str, str]:
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
                encoded = "".join(raw)
                return json.loads(encoded), encoded

    def read_scalar(self) -> Any:
        self.skip_ws()
        if self.peek() == '"':
            return self.read_string()[0]
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
                raise ValueError("unterminated selected row")
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
            raise ValueError("unbalanced selected row")
        result = json.loads("".join(capture))
        self.seek(cursor)
        return result


def _read_row(stream: JsonStream) -> tuple[str, dict | None]:
    start = stream.mark()
    stream.expect("{")
    family = None
    while True:
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        key, _ = stream.read_string()
        stream.skip_ws()
        stream.expect(":")
        stream.skip_ws()
        if key == "family":
            if family is not None:
                raise ValueError("confirmation row has duplicate family field")
            family, _ = stream.read_string()
        else:
            stream.skip_value()
        stream.skip_ws()
        if stream.peek() == "}":
            stream.get()
            break
        stream.expect(",")
    if family is None:
        raise ValueError("confirmation row lacks family field")
    return (family, stream.read_object_at(start) if family == NEW_FAMILY else None)


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
    metadata: dict = {}
    new_rows: list[dict] = []
    skipped_families: Counter[str] = Counter()
    try:
        stream.skip_ws()
        stream.expect("{")
        while True:
            stream.skip_ws()
            if stream.peek() == "}":
                stream.get()
                break
            key, _ = stream.read_string()
            stream.skip_ws()
            stream.expect(":")
            stream.skip_ws()
            if key == "rows":
                stream.expect("[")
                stream.skip_ws()
                if stream.peek() != "]":
                    while True:
                        family, row = _read_row(stream)
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
        raise ValueError("confirmation report metadata differs from the frozen PLAN-N1-102 receipt")
    if skipped != {PRIOR_FAMILY: 180}:
        raise ValueError("exactly 180 prior-family rows must be skipped without decoding")


def compare_pair(position_rows: list[dict], control_rows: list[dict]) -> dict:
    if len(position_rows) != len(control_rows):
        raise ValueError("paired arms have unequal item counts")
    position_by_id = {row["id"]: row for row in position_rows}
    control_by_id = {row["id"]: row for row in control_rows}
    if len(position_by_id) != len(position_rows) or len(control_by_id) != len(control_rows):
        raise ValueError("duplicate item ID in a paired arm")
    if position_by_id.keys() != control_by_id.keys():
        raise ValueError("paired arms do not contain identical item IDs")
    counts = Counter()
    for item_id in position_by_id:
        position = position_by_id[item_id]
        control = control_by_id[item_id]
        position_exact = position.get("free_exact")
        control_exact = control.get("free_exact")
        if type(position_exact) is not bool or type(control_exact) is not bool:
            raise ValueError("paired rows need boolean free_exact fields")
        position_output = position.get("output_hex")
        control_output = control.get("output_hex")
        if not isinstance(position_output, str) or not isinstance(control_output, str):
            raise ValueError("paired rows need hex output fields")
        try:
            position_bytes = bytes.fromhex(position_output)
            control_bytes = bytes.fromhex(control_output)
        except ValueError as exc:
            raise ValueError("paired output_hex field is invalid") from exc
        if position_exact and control_exact:
            counts["both_correct"] += 1
        elif position_exact:
            counts["position_only_correct"] += 1
        elif control_exact:
            counts["control_only_correct"] += 1
        else:
            counts["both_wrong"] += 1
        counts["output_bytes_identical"] += position_bytes == control_bytes
    counts["denominator"] = len(position_rows)
    if sum(
        counts[key]
        for key in ("both_correct", "position_only_correct", "control_only_correct", "both_wrong")
    ) != len(position_rows):
        raise AssertionError("paired correctness categories do not partition their denominator")
    keys = (
        "both_correct",
        "position_only_correct",
        "control_only_correct",
        "both_wrong",
        "output_bytes_identical",
    )
    return {**{key: counts[key] for key in keys}, "denominator": len(position_rows)}


def analyze_rows(rows: list[dict]) -> dict:
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    cells: dict[tuple[int, str], list[dict]] = {key: [] for key in expected}
    seen = set()
    for row in rows:
        if row.get("family") != NEW_FAMILY:
            raise ValueError("analysis input must contain only the v9 new family")
        key = (row.get("seed"), row.get("arm"))
        if key not in expected:
            raise ValueError("unexpected seed or arm in v9 attribution rows")
        identity = (*key, row.get("id"))
        if identity in seen:
            raise ValueError("duplicate seed/arm/item attribution row")
        seen.add(identity)
        if row.get("kind") not in KINDS:
            raise ValueError("unexpected item kind in v9 attribution rows")
        cells[key].append(row)
    if len(rows) != 180 or any(len(values) != 20 for values in cells.values()):
        raise ValueError("attribution needs exactly twenty rows for each of nine new endpoints")

    ids_by_endpoint = {key: {row["id"] for row in values} for key, values in cells.items()}
    if len({frozenset(ids) for ids in ids_by_endpoint.values()}) != 1:
        raise ValueError("all new endpoints must use the same twenty confirmation item IDs")

    comparisons = {}
    for control in ("cut", "terminal"):
        per_seed_kind = {}
        per_seed = {}
        per_kind = {}
        for seed in SEEDS:
            seed_position = cells[seed, "position"]
            seed_control = cells[seed, control]
            per_seed_results = []
            for kind in KINDS:
                position_rows = [row for row in seed_position if row["kind"] == kind]
                control_rows = [row for row in seed_control if row["kind"] == kind]
                if len(position_rows) != 5 or len(control_rows) != 5:
                    raise ValueError("each seed/arm/kind cell must contain five rows")
                result = compare_pair(position_rows, control_rows)
                per_seed_kind[f"{seed}_{kind}"] = result
                per_seed_results.append(result)
                per_kind.setdefault(kind, []).append(result)
            per_seed[seed] = _sum_cells(per_seed_results)
        per_kind_totals = {kind: _sum_cells(values) for kind, values in per_kind.items()}
        overall = _sum_cells(list(per_seed_kind.values()))
        comparisons[f"position_vs_{control}"] = {
            "per_seed_kind": per_seed_kind,
            "per_seed": per_seed,
            "per_kind": per_kind_totals,
            "overall": overall,
        }
    return {
        "rows_analyzed": len(rows),
        "seed_count": len(SEEDS),
        "arms_compared": ["position_vs_cut", "position_vs_terminal"],
        "per_kind_denominator_per_seed_pair": 5,
        "per_seed_denominator_per_arm_pair": 20,
        "overall_denominator_per_arm_pair": 60,
        "comparisons": comparisons,
    }


def _sum_cells(cells: list[dict]) -> dict:
    keys = (
        "both_correct",
        "position_only_correct",
        "control_only_correct",
        "both_wrong",
        "output_bytes_identical",
        "denominator",
    )
    return {key: sum(cell.get(key, 0) for cell in cells) for key in keys}


def write_new(path: Path, value: dict) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def run_attribution() -> dict:
    if REPORT.exists():
        raise FileExistsError("PLAN-N1-103 attribution report is one-shot")
    if sha256(CONFIRMATION) != CONFIRMATION_SHA256:
        raise ValueError("v9 confirmation report bytes differ from the sealed source")
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-103 bytes differ from the approved frozen protocol")
    started = time.process_time()
    metadata, rows, skipped = stream_confirmation(CONFIRMATION)
    validate_confirmation_metadata(metadata, skipped)
    attribution = analyze_rows(rows)
    cpu_seconds = time.process_time() - started
    report = {
        "schema": "n1-hard-neighbor-v9-item-error-attribution-v1",
        "status": "complete" if cpu_seconds <= CPU_LIMIT_SECONDS else "stopped",
        "complete": cpu_seconds <= CPU_LIMIT_SECONDS,
        "exploratory_only": True,
        "causal_claim_allowed": False,
        "gate_recomputed": False,
        "v9_confirmation_report_sha256": sha256(CONFIRMATION),
        "plan_sha256": sha256(PLAN),
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "new_family_rows_decoded": len(rows),
        "prior_family_rows_skipped_without_decoding": skipped.get(PRIOR_FAMILY, 0),
        "metadata_and_sources_unchanged": True,
        "model_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "new_face_queries": 0,
        "can_promote": False,
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(cpu_seconds, 6),
        "attribution": attribution,
        "interpretation_limit": (
            "descriptive overlap among consumed v9 rows only; no causal, capability, C3, C4, "
            "retention, three-organ, or promotion claim"
        ),
    }
    write_new(REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    args = parser.parse_args()
    del args
    report = run_attribution()
    print(
        json.dumps(
            {
                "status": report["status"],
                "complete": report["complete"],
                "report": str(REPORT),
                "report_sha256": sha256(REPORT),
                "rows_analyzed": report["new_family_rows_decoded"],
                "cpu_seconds": report["cpu_seconds"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
