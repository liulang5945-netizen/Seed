"""SPEC-N1-20 frozen train-only content interventions and read diagnostics."""

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
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402

BASELINE = ROOT / "reports/taiji_n1_v3_train_calibration_20261015.json"
BASELINE_SHA = "ab7cfe7c007c4fc2f9f6a13903bc31abb7714ba02f711949fb41a0e5d443eb20"
CHECKPOINT_SHA = "9ec27cbd1d353ca5c2ca3c88796aaad9a78efe5787232b6731dd534a09d0b1f6"


def select(manifest):
    train = calibration.training_items(manifest)
    selected = [item for item in train if item["kind"] in ("lookup", "revision")]
    if len(selected) != 96:
        raise ValueError("all 96 material training items required")
    return selected


def variants(item, replacements):
    if item["split"] != "train" or item["kind"] not in ("lookup", "revision"):
        raise ValueError("material training episode required before any model access")
    suffix = f" uses {item['answer']}."
    indices = [i for i, text in enumerate(item["observations"]) if text.endswith(suffix)]
    if len(indices) != 1 or len(item["observations"]) != 2:
        raise ValueError("unique gold material source required")
    target = indices[0]
    changed = {}
    for name, index, replacement in (
        ("target", target, replacements[0]),
        ("distractor", 1 - target, replacements[1]),
    ):
        text = item["observations"][index]
        previous = text.removesuffix(".").split()[-1]
        if len(replacement) != len(previous) or replacement == previous:
            raise ValueError("intervention requires a different equal-length value")
        variant = deepcopy(item)
        variant["observations"][index] = text[: -len(previous) - 1] + replacement + "."
        if name == "target":
            variant["answer"] = replacement
        changed[name] = variant
    return target, changed


@torch.no_grad()
def infer(model, item, session):
    if item["split"] != "train":
        raise ValueError("no dev or holdout model queries")
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = model.evidence(session)
    logits, _, _ = model.step(tape, session, model.question_state(data.question), 256)
    trace = model.generate_trace(tape, session, data.question, 64)
    return (
        {
            "raw_hex": trace.raw.hex(),
            "stop_reason": trace.stop_reason,
            "generated_steps": trace.generated_steps,
            "exact": trace.raw == item["answer"].encode() and trace.stop_reason == "boundary",
        },
        torch.softmax(logits, dim=-1),
        tape,
    )


@torch.no_grad()
def read_profile(model, item, tape, session, target_source):
    if item["split"] != "train":
        raise ValueError("no heldout teacher diagnostics")
    target = item["answer"].encode()
    logits, weights = model.teacher_logits(tape, session, item["question"].encode(), target)
    labels = torch.tensor([*target, 256])
    text = item["observations"][target_source].encode()
    local = text.rfind(target)
    if local < 0:
        raise ValueError("target material span absent")
    source_name = f"source-{target_source + 1}"
    source_positions = [i for i, (source, _) in enumerate(tape.positions) if source == source_name]
    span = [
        i
        for i, (source, pos) in enumerate(tape.positions)
        if source == source_name and local <= pos < local + len(target)
    ]
    answer_weights = weights[: len(target)]
    return {
        "teacher_tokens_correct": int((logits.argmax(dim=-1) == labels).sum()),
        "teacher_tokens": len(labels),
        "teacher_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
        "target_source_mass": float(answer_weights[:, source_positions].sum(dim=-1).mean()),
        "target_span_mass": float(answer_weights[:, span].sum(dim=-1).mean()),
        "matching_byte_address_mass": float(
            answer_weights[torch.arange(len(target)), torch.tensor(span)].mean()
        ),
        "peak_in_target_span_fraction": float(
            torch.isin(answer_weights.argmax(dim=-1), torch.tensor(span)).float().mean()
        ),
        "uniform_source_mass": len(source_positions) / len(tape.positions),
        "uniform_span_mass": len(span) / len(tape.positions),
    }


def summarize(rows):
    result = {}
    for kind in ("lookup", "revision"):
        selected = [row for row in rows if row["kind"] == kind]
        n = len(selected)
        result[kind] = {
            "items": n,
            **{
                f"{variant}_exact": sum(row[variant]["exact"] for row in selected)
                for variant in ("original", "target", "distractor")
            },
            **{
                f"{variant}_output_changed": sum(
                    (row[variant]["raw_hex"], row[variant]["stop_reason"])
                    != (row["original"]["raw_hex"], row["original"]["stop_reason"])
                    for row in selected
                )
                for variant in ("target", "distractor")
            },
            **{
                f"mean_{variant}_first_tv": (
                    sum(row[f"{variant}_first_tv"] for row in selected) / n if n else None
                )
                for variant in ("target", "distractor")
            },
            **{
                f"mean_{field}": (sum(row["read"][field] for row in selected) / n if n else None)
                for field in (
                    "teacher_ce",
                    "target_source_mass",
                    "target_span_mass",
                    "matching_byte_address_mass",
                    "peak_in_target_span_fraction",
                    "uniform_source_mass",
                    "uniform_span_mass",
                )
            },
            "teacher_token_accuracy": (
                sum(row["read"]["teacher_tokens_correct"] for row in selected)
                / sum(row["read"]["teacher_tokens"] for row in selected)
                if n
                else None
            ),
        }
    return result


def audit():
    report = ROOT / "reports/taiji_n1_train_content_read_20261015.json"
    if report.exists():
        raise FileExistsError("content diagnostic exists; no repeat or overwrite")
    if (
        paired.sha(BASELINE) != BASELINE_SHA
        or paired.sha(calibration.MANIFEST) != calibration.MANIFEST_SHA
    ):
        raise ValueError("frozen source report or manifest changed")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    checks = [check for check in baseline["restore_checks"] if check["path"].endswith("final.pt")]
    if len(checks) != 1 or checks[0]["sha256"] != CHECKPOINT_SHA or not baseline["complete"]:
        raise ValueError("complete frozen calibration final required")
    checkpoint = Path(checks[0]["path"])
    if paired.sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("calibration source checkpoint changed")
    manifest = json.loads(calibration.MANIFEST.read_text(encoding="utf-8"))
    selected = select(manifest)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != calibration.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("unexpected calibration protocol or updates")
    model = NativeEvidenceCandidate.restore_bundle(payload["candidate"])
    before = paired.fingerprint(model.state_dict())
    identity = {
        "calibration": calibration.code_identity(),
        "audit_sha256": paired.sha(Path(__file__)),
    }
    used = {
        text.removesuffix(".").split()[-1]
        for item in calibration.training_items(manifest)
        for text in item["observations"]
        if " uses " in text
    }
    rng = random.Random(20)

    def novel():
        while True:
            value = "".join(rng.choices(string.ascii_lowercase, k=6))
            if value not in used:
                used.add(value)
                return value

    result = {
        "schema": "n1-train-content-read-diagnostic-v1",
        "baseline_sha256": BASELINE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "manifest_sha256": calibration.MANIFEST_SHA,
        "code_identity": identity,
        "rows": [],
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "free_queries": 0,
        "teacher_queries": 0,
        "cpu_limit_seconds": 60,
        "can_promote": False,
        "status": "running",
    }
    try:
        for item in selected:
            if time.process_time() >= 58:
                raise paired.BudgetStop("content diagnostic CPU reserve reached")
            replacements = (novel(), novel())
            source, changed = variants(item, replacements)
            session = f"content-original-{item['id']}"
            original, probabilities, tape = infer(model, item, session)
            result["free_queries"] += 1
            row = {"id": item["id"], "kind": item["kind"], "original": original}
            row["read"] = read_profile(model, item, tape, session, source)
            result["teacher_queries"] += 1
            row["replacements"] = dict(zip(("target", "distractor"), replacements, strict=True))
            for name, variant in changed.items():
                prediction, new_probabilities, _ = infer(
                    model, variant, f"content-{name}-{item['id']}"
                )
                result["free_queries"] += 1
                row[name] = prediction
                row[f"{name}_first_tv"] = float((new_probabilities - probabilities).abs().sum() / 2)
            result["rows"].append(row)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result.update(
            cpu_seconds=time.process_time(),
            parameters_unchanged=before == paired.fingerprint(model.state_dict()),
            checkpoint_unchanged=paired.sha(checkpoint) == CHECKPOINT_SHA,
            manifest_unchanged=paired.sha(calibration.MANIFEST) == calibration.MANIFEST_SHA,
            code_unchanged=(
                identity
                == {
                    "calibration": calibration.code_identity(),
                    "audit_sha256": paired.sha(Path(__file__)),
                }
            ),
            summary=summarize(result["rows"]),
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 96
            and result["free_queries"] == 288
            and result["teacher_queries"] == 96
            and result["cpu_seconds"] <= 60
            and all(
                result[key]
                for key in (
                    "parameters_unchanged",
                    "checkpoint_unchanged",
                    "manifest_unchanged",
                    "code_unchanged",
                )
            )
        )
        paired.write_new(report, result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = audit()
    print(json.dumps({key: output[key] for key in ("complete", "cpu_seconds", "summary")}))
    raise SystemExit(0 if output["complete"] else 1)
