"""PLAN125: static mechanism map and error taxonomy; no model imports or queries."""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports/taiji_n1_existing_binding_error_audit_20261019.json"
INPUTS = {
    "v13": (
        ROOT / "plans/manifests/n1_hard_neighbor_v13_confirmation_seed113_20261019.json",
        ROOT / "reports/taiji_n1_v13_after_n122_confirmation_20261019.json",
    ),
    "v14": (
        ROOT / "plans/manifests/n1_hard_neighbor_v14_confirmation_seed114_20261019.json",
        ROOT / "reports/taiji_n1_v14_after_n123_confirmation_20261019.json",
    ),
}
MECHANISMS = (
    "taiji/native_local_question_evidence_candidate.py",
    "taiji/native_contextual_question_evidence_candidate.py",
    "taiji/native_residual_question_evidence_candidate.py",
    "taiji/native_question_aligned_evidence_candidate.py",
    "taiji/native_local_relation_evidence_candidate.py",
    "taiji/native_relative_evidence_candidate.py",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(item: dict, row: dict) -> str:
    output = bytes.fromhex(row["output_hex"])
    exact = output == item["answer"].encode()
    if exact != row["free_exact"]:
        raise ValueError(f"stored exact flag differs from bytes: {item['id']}")
    if exact:
        return "correct"
    if output == b"unknown":
        return "false_abstention"
    if item["kind"] == "transition" and output in (b"north", b"east", b"south", b"west"):
        return "wrong_valid_direction"
    subject = item["entities"][0]
    material_rows = [line for line in item["observations"] if " uses " in line]
    matching = [
        line for line in material_rows if line.removesuffix(".").split()[-1].encode() == output
    ]
    if any(not line.startswith(subject + " ") for line in matching):
        return "non_target_material"
    if matching:
        return "stale_target_material"
    return "other_output"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    started = time.process_time()
    paths = [Path(__file__).resolve()]
    summaries = {}
    for face_name, (face_path, result_path) in INPUTS.items():
        face = json.loads(face_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not result["complete"]
            or result["confirmation_queries"] != 360
            or result["face_sha256"] != digest(face_path)
        ):
            raise ValueError(f"incomplete or mismatched report: {face_name}")
        items = {
            item["id"]: item
            for item in face["episodes"]
            if item["id"] in face["subsets"]["confirm"]
        }
        summaries[face_name] = {}
        for cohort in ("baseline", "candidate"):
            rows = result[f"{cohort}_rows"]
            if (
                len(rows) != 180
                or len({(row["seed"], row["arm"], row["id"]) for row in rows}) != 180
            ):
                raise ValueError("confirmation row identities incomplete")
            summaries[face_name][cohort] = {}
            for arm in ("position", "cut", "terminal"):
                summaries[face_name][cohort][arm] = {}
                for kind in ("lookup", "revision", "transition", "unknown"):
                    selected = [row for row in rows if row["arm"] == arm and row["kind"] == kind]
                    if len(selected) != 15:
                        raise ValueError("class denominator differs")
                    summaries[face_name][cohort][arm][kind] = {
                        "items": 15,
                        "outputs": dict(
                            Counter(classify(items[row["id"]], row) for row in selected)
                        ),
                        "free_teacher_exact_disagreements": sum(
                            row["free_exact"] != row["teacher_exact"] for row in selected
                        ),
                    }
        paths.extend((face_path, result_path))
    definitions = {}
    for raw_path in MECHANISMS:
        path = ROOT / raw_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions[raw_path] = [
            {"function": node.name, "start": node.lineno, "end": node.end_lineno}
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name
            in (
                "bind_question",
                "condition",
                "context_sequence",
                "relation_inputs",
                "relation_tape",
                "step",
            )
        ]
        paths.append(path)
    report = {
        "schema": "n1-plan125-existing-binding-error-audit-v1",
        "status": "complete",
        "source_hashes": {str(path.resolve()): digest(path) for path in paths},
        "mechanism_definitions": definitions,
        "taxonomy": summaries,
        "model_loads": 0,
        "checkpoint_loads": 0,
        "model_queries": 0,
        "parameter_updates": 0,
        "dev_queries": 0,
        "can_promote": False,
        "cpu_seconds": time.process_time() - started,
        "limits": "Output categories describe saved errors; they do not identify a causal mechanism. Existing code already conditions question representations and key/value tape on question bytes.",
    }
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
    print(
        json.dumps({"status": "complete", "model_queries": 0, "cpu_seconds": report["cpu_seconds"]})
    )


if __name__ == "__main__":
    main()
