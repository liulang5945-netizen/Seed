"""Approved PLAN42: train-only frozen entity counterfactuals, zero updates."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training import run_taiji_n1_reader_transfer as source  # noqa: E402
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from taiji.native_residual_question_evidence_candidate import (
    ResidualQuestionEvidenceCandidate,
)  # noqa: E402

SOURCE_SHA = "fe50c400f12d2258979238ee00105ff09e6131d7548640b3de273f34ba50f9d5"
MANIFEST = ROOT / "plans/manifests/n1_entity_binding_diagnostic_20261015.json"
REPORT = ROOT / "reports/taiji_n1_transfer_binding_diagnostic_v2_20261015.json"
CONDITIONS = ("original", "rename", "query_switch")


def validate_inputs(inputs, train, pairs):
    originals = {e["id"]: e for e in train if e["kind"] in ("lookup", "unknown")}
    paired_items = {e["parent_id"]: e for e in pairs}
    if len(inputs) != 96 or {r["id"] for r in inputs} != set(originals):
        raise ValueError("96 unique original train interventions required")
    for row in inputs:
        if set(row["variants"]) != set(CONDITIONS):
            raise ValueError("all three frozen conditions required")
        if row["variants"]["original"] != originals[row["id"]]:
            raise ValueError("original train inputs changed")
        for name in CONDITIONS:
            if row["variants"][name]["split"] != "train":
                raise ValueError("only train variants allowed")
        switched = row["variants"]["query_switch"]
        expected = paired_items[row["id"]]
        if any(switched[k] != expected[k] for k in ("question", "answer", "observations")):
            raise ValueError("switch must reproduce the trained pair")
    return inputs


def checked_payload(payload):
    if payload["protocol"] != source.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fixed single-model transferred checkpoint protocol required")
    if payload["protocol"]["seed"] != 11 or payload["protocol"]["arm"] != "position":
        raise ValueError("fixed seed11 position required")
    return payload


def identity():
    return {
        "diagnostic": paired.sha(Path(__file__)),
        "source": source.code_identity(),
        "measurement": paired.sha(
            ROOT / "scripts/training/audit_taiji_n1_native_evidence_train_fit.py"
        ),
    }


@torch.no_grad()
def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive diagnostic artifacts exist; no repeat")
    if (
        paired.sha(source.REPORT) != SOURCE_SHA
        or paired.sha(calibration.MANIFEST) != calibration.MANIFEST_SHA
    ):
        raise ValueError("frozen input evidence changed")
    baseline = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if (
        not baseline["complete"]
        or not baseline["training_sufficient"]
        or baseline["code_identity"] != source.code_identity()
    ):
        raise ValueError("complete frozen continuation required")
    entries = [
        dict(e, seed=11, arm="position")
        for e in baseline["restore_checks"]
        if Path(e["path"]).name == "final.pt"
    ]
    if len(entries) != 1:
        raise ValueError("unique fixed seed11 transferred final required")
    if paired.sha(MANIFEST) != "572a270bcb5e292baee7577b5c0ec2d3f313bd93ae338fa6d5984a646a2fe9c6":
        raise ValueError("frozen original intervention manifest changed")
    inputs = json.loads(MANIFEST.read_text(encoding="utf-8"))["rows"]
    if len(inputs) != 96 or {r["id"] for r in inputs} != {
        e["id"]
        for e in calibration.training_items(
            json.loads(calibration.MANIFEST.read_text(encoding="utf-8"))
        )
        if e["kind"] in ("lookup", "unknown")
    }:
        raise ValueError("all original paired train intervention identities required")
    train = source.checked_train()
    validate_inputs(inputs, train, source.checked_pairs(train))
    manifest_sha = paired.sha(MANIFEST)
    prior_path = ROOT / "reports/taiji_n1_transfer_binding_diagnostic_20261015.json"
    prior_sha = "752cef26dae183bd9f0a3bf4046f3fd118b96b8a26c6e3993240c6955957f848"
    if paired.sha(prior_path) != prior_sha:
        raise ValueError("prior zero-query rejection changed")
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if prior["complete"] or prior["teacher_queries"] or prior["free_queries"] or prior["rows"]:
        raise ValueError("only pre-query schema rejection can be corrected in v2")
    frozen_code = identity()
    result = dict(
        schema="n1-transfer-binding-diagnostic-v1",
        source_sha256=SOURCE_SHA,
        manifest_sha256=manifest_sha,
        code_identity=frozen_code,
        parameter_updates=0,
        dev_queries=0,
        holdout_queries=0,
        teacher_queries=0,
        free_queries=0,
        cpu_limit_seconds=60,
        rows=[],
        status="running",
        can_promote=False,
    )
    guards = []
    try:
        for entry in entries:
            path = Path(entry["path"])
            if paired.sha(path) != entry["sha256"]:
                raise ValueError("frozen checkpoint changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            checked_payload(payload)
            model = ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
            before = paired.fingerprint(model.bundle())
            original = {
                r["id"]: r
                for r in baseline["rows"]
                if r["completed_updates"] == 6144 and r["evaluation_set"] == "original"
            }
            switched = {
                r["id"].removeprefix("pair-"): r
                for r in baseline["rows"]
                if r["completed_updates"] == 6144 and r["evaluation_set"] == "pair"
            }
            for item in inputs:
                predictions = {}
                for condition in CONDITIONS:
                    if time.process_time() + prior["cpu_seconds"] >= 58:
                        raise paired.BudgetStop("diagnostic reserve reached")
                    session = f"binding-{entry['seed']}-{item['id']}-{condition}"
                    variant = item["variants"][condition]
                    try:
                        row = source.measure(model, variant, "position", session)
                        result["teacher_queries"] += 1
                        result["free_queries"] += 1
                    finally:
                        model._observations.pop(session, None)
                    if condition in ("original", "query_switch"):
                        old = (original if condition == "original" else switched)[item["id"]]
                        if (
                            any(
                                row[k] != old[k]
                                for k in ("raw_hex", "stop_reason", "teacher_tokens_correct")
                            )
                            or abs(row["teacher_ce"] - old["teacher_ce"]) > 1e-6
                        ):
                            raise ValueError("original control differs from frozen train")
                    raw = bytes.fromhex(row["raw_hex"])
                    values = [
                        o.removesuffix(".").split()[-1].encode() for o in variant["observations"]
                    ]
                    row.update(
                        seed=entry["seed"],
                        condition=condition,
                        category=(
                            "gold"
                            if row["free_exact"]
                            else (
                                "observed_value"
                                if raw in values
                                else "unknown" if raw == b"unknown" else "other"
                            )
                        ),
                    )
                    predictions[condition] = row
                    result["rows"].append(row)
                for row in predictions.values():
                    row["same_as_original"] = (row["raw_hex"], row["stop_reason"]) == (
                        predictions["original"]["raw_hex"],
                        predictions["original"]["stop_reason"],
                    )
            guards.append(before == paired.fingerprint(model.bundle()))
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    result.update(
        cpu_seconds=time.process_time() + prior["cpu_seconds"],
        prior_rejection_sha256=prior_sha,
        prior_rejection_cpu_seconds=prior["cpu_seconds"],
        parameters_and_sources_unchanged=len(guards) == 1 and all(guards),
        inputs_unchanged=paired.sha(source.REPORT) == SOURCE_SHA
        and paired.sha(calibration.MANIFEST) == calibration.MANIFEST_SHA
        and paired.sha(MANIFEST) == manifest_sha
        and all(paired.sha(Path(e["path"])) == e["sha256"] for e in entries),
        code_unchanged=identity() == frozen_code,
    )
    result["summary"] = {
        kind: {
            condition: {
                key: sum(
                    r[key]
                    for r in result["rows"]
                    if r["kind"] == kind and r["condition"] == condition
                )
                for key in ("free_exact", "same_as_original")
            }
            for condition in CONDITIONS
        }
        for kind in ("lookup", "unknown")
    }
    result["complete"] = (
        result["status"] == "complete"
        and len(result["rows"]) == 288
        and result["teacher_queries"] == result["free_queries"] == 288
        and result["cpu_seconds"] <= 60
        and all(
            result[k]
            for k in ("parameters_and_sources_unchanged", "inputs_unchanged", "code_unchanged")
        )
    )
    paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = audit()
    print(json.dumps({k: output[k] for k in ("complete", "cpu_seconds", "summary")}))
    raise SystemExit(0 if output["complete"] else 1)
