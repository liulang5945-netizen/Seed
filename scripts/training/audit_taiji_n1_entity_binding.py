"""Approved PLAN34: train-only frozen entity counterfactuals, zero updates."""

from __future__ import annotations

import json
import random
import string
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training import run_taiji_n1_relative_continuation as source  # noqa: E402
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate  # noqa: E402

SOURCE_SHA = "871f054fcff6a72f713849de2a8b3b1ae6680e4581bccad317fef65a3c133045"
MANIFEST = ROOT / "plans/manifests/n1_entity_binding_diagnostic_20261015.json"
REPORT = ROOT / "reports/taiji_n1_entity_binding_diagnostic_20261015.json"
CONDITIONS = ("original", "rename", "query_switch")


def variants(item, forbidden, rng):
    if item["split"] != "train" or item["kind"] not in ("lookup", "unknown"):
        raise ValueError("only train lookup/unknown allowed")
    names = item["entities"]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("three unique entities required")
    mapping = {}
    for name in names:
        while True:
            replacement = "".join(rng.choices(string.ascii_lowercase, k=10))
            if replacement not in forbidden:
                break
        forbidden.add(replacement)
        mapping[name] = replacement
    renamed = deepcopy(item)
    renamed["entities"] = [mapping[n] for n in names]
    for field in ("observations", "question"):
        texts = item[field] if field == "observations" else [item[field]]
        rewritten = []
        for text in texts:
            for old, new in mapping.items():
                text = text.replace(old, new)
            rewritten.append(text)
        renamed[field] = rewritten if field == "observations" else rewritten[0]
    switched = deepcopy(item)
    old = item["question"].removeprefix("material of ").removesuffix("?")
    observed = [text.split()[0] for text in item["observations"]]
    if item["kind"] == "lookup":
        choices = [i for i, name in enumerate(observed) if name != old]
        if old not in observed or len(choices) != 1:
            raise ValueError("lookup requires distinct observed entities")
        index = choices[0]
    else:
        if old in observed:
            raise ValueError("unknown question must be absent")
        index = 0
    switched["question"] = f"material of {observed[index]}?"
    switched["answer"] = item["observations"][index].removesuffix(".").split()[-1]
    return dict(original=deepcopy(item), rename=renamed, query_switch=switched)


def build(manifest):
    forbidden = {n for e in manifest["episodes"] for n in e["entities"]}
    rng = random.Random(34)
    selected = [
        e for e in calibration.training_items(manifest) if e["kind"] in ("lookup", "unknown")
    ]
    if len(selected) != 96 or len({e["id"] for e in selected}) != 96:
        raise ValueError("96 unique train items required")
    return [
        {"id": e["id"], "kind": e["kind"], "variants": variants(e, forbidden, rng)}
        for e in selected
    ]


def identity():
    return {
        "diagnostic": paired.sha(Path(__file__)),
        "source": source.identity(),
        "measurement": paired.sha(Path(source.source.__file__)),
    }


@torch.no_grad()
def audit():
    if REPORT.exists() or MANIFEST.exists():
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
        or baseline["identity"] != source.identity()
    ):
        raise ValueError("complete frozen continuation required")
    entries = [e for e in baseline["frozen_models"] if e["arm"] == "position"]
    if len(entries) != 3 or {e["seed"] for e in entries} != set(paired.SEEDS):
        raise ValueError("three unique frozen position models required")
    inputs = build(json.loads(calibration.MANIFEST.read_text(encoding="utf-8")))
    paired.write_new(
        MANIFEST,
        {"schema": "n1-entity-binding-variants-v1", "source_sha256": SOURCE_SHA, "rows": inputs},
    )
    manifest_sha = paired.sha(MANIFEST)
    frozen_code = identity()
    result = dict(
        schema="n1-entity-binding-diagnostic-v1",
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
            if (
                payload["protocol"] != source.PROTOCOL
                or payload["completed_updates"] != 9216
                or payload["arm"] != "position"
                or payload["seed"] != entry["seed"]
            ):
                raise ValueError("checkpoint protocol mismatch")
            model = RelativeEvidenceCandidate.restore_bundle(payload["candidate"])
            before = paired.fingerprint(model.bundle())
            original = {
                r["id"]: r
                for r in baseline["train_rows"]
                if r["seed"] == entry["seed"] and r["arm"] == "position"
            }
            for item in inputs:
                predictions = {}
                for condition in CONDITIONS:
                    if time.process_time() >= 58:
                        raise paired.BudgetStop("diagnostic reserve reached")
                    session = f"binding-{entry['seed']}-{item['id']}-{condition}"
                    variant = item["variants"][condition]
                    try:
                        row = source.source.measure(model, variant, "position", session)
                        result["teacher_queries"] += 1
                        result["free_queries"] += 1
                    finally:
                        model._observations.pop(session, None)
                    if condition == "original":
                        old = original[item["id"]]
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
        cpu_seconds=time.process_time(),
        parameters_and_sources_unchanged=len(guards) == 3 and all(guards),
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
        and len(result["rows"]) == 864
        and result["teacher_queries"] == result["free_queries"] == 864
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
