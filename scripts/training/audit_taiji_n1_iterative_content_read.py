"""SPEC26: identical train interventions after approved two-hop calibration."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import audit_taiji_n1_train_content_read as previous  # noqa: E402
from scripts.training import run_taiji_n1_iterative_read as fidelity  # noqa: E402
from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from taiji.native_evidence_fidelity import observation_logits  # noqa: E402
from taiji.native_iterative_evidence_candidate import IterativeEvidenceCandidate  # noqa: E402

SOURCE_SHA = "874d06d0452b7e09305232eed3c7d7d95a62f9de82b1971ee35bc1f49f1dff4b"
CHECKPOINT_SHA = "40038475198ec67ec6908ef2a3d37d0303b02d710c841ed35acba96c64fafc7f"
OLD_REPORT = ROOT / "reports/taiji_n1_train_content_read_20261015.json"
OLD_SHA = "98d55cae330e80eadfb929f7257b6dd287de6eb87b6e5bc7badfcd30cc2e3e72"


def paired_inputs(manifest, rows):
    selected = previous.select(manifest)
    replacements = {row["id"]: row["replacements"] for row in rows}
    if len(rows) != 96 or set(replacements) != {item["id"] for item in selected}:
        raise ValueError("all 96 unique original intervention identities required")
    return [
        (item, (replacements[item["id"]]["target"], replacements[item["id"]]["distractor"]))
        for item in selected
    ]


@torch.no_grad()
def reconstruct(model, item, session, tape, source_index, word):
    if item["split"] != "train":
        raise ValueError("fidelity read diagnostic requires train episodes")
    logits, labels = observation_logits(model, tape, session)
    text = item["observations"][source_index].encode()
    start = text.rfind(word.encode())
    if start < 0:
        raise ValueError("replacement span missing from external observation")
    span = [
        i
        for i, (source, offset) in enumerate(tape.positions)
        if source == f"source-{source_index + 1}" and start <= offset < start + len(word)
    ]
    correct = logits.argmax(dim=-1) == labels
    return {
        "observation_correct": int(correct.sum()),
        "observation_tokens": len(labels),
        "span_correct": int(correct[span].sum()),
        "span_tokens": len(span),
    }


def code_identity():
    return {
        "fidelity": fidelity.code_identity(),
        "original_diagnostic_sha256": paired.sha(Path(previous.__file__)),
        "audit_sha256": paired.sha(Path(__file__)),
    }


def audit():
    report = ROOT / "reports/taiji_n1_iterative_content_read_20261015.json"
    if report.exists():
        raise FileExistsError("fidelity content diagnostic exists; no repeat or overwrite")
    if paired.sha(fidelity.REPORT) != SOURCE_SHA or paired.sha(OLD_REPORT) != OLD_SHA:
        raise ValueError("frozen source evidence changed")
    baseline = json.loads(fidelity.REPORT.read_text(encoding="utf-8"))
    old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    if not baseline["complete"] or baseline["protocol"] != fidelity.PROTOCOL or not old["complete"]:
        raise ValueError("complete source protocols required")
    checks = [check for check in baseline["restore_checks"] if check["path"].endswith("final.pt")]
    if len(checks) != 1 or checks[0]["sha256"] != CHECKPOINT_SHA:
        raise ValueError("unique frozen fidelity final required")
    checkpoint = Path(checks[0]["path"])
    if (
        paired.sha(checkpoint) != CHECKPOINT_SHA
        or paired.sha(calibration.MANIFEST) != calibration.MANIFEST_SHA
    ):
        raise ValueError("frozen fidelity checkpoint or manifest changed")
    manifest = json.loads(calibration.MANIFEST.read_text(encoding="utf-8"))
    inputs = paired_inputs(manifest, old["rows"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["protocol"] != fidelity.PROTOCOL or payload["completed_updates"] != 6144:
        raise ValueError("fidelity checkpoint protocol mismatch")
    model = IterativeEvidenceCandidate.restore_bundle(payload["candidate"])
    before = paired.fingerprint(model.state_dict())
    original_rows = {row["id"]: row for row in baseline["rows"] if row["completed_updates"] == 6144}
    result = {
        "schema": "n1-iterative-content-read-v1",
        "source_sha256": SOURCE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "old_intervention_report_sha256": OLD_SHA,
        "manifest_sha256": calibration.MANIFEST_SHA,
        "code_identity": code_identity(),
        "read_hops": 2,
        "attention_profile": "second-hop-only",
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "free_queries": 0,
        "teacher_queries": 0,
        "reconstruction_queries": 0,
        "can_promote": False,
        "rows": [],
        "cpu_limit_seconds": 60,
        "status": "running",
    }
    try:
        for item, replacements in inputs:
            if time.process_time() >= 58:
                raise paired.BudgetStop("fidelity content read CPU reserve reached")
            source, changed = previous.variants(item, replacements)
            session = f"iterative-read-original-{item['id']}"
            prediction, probabilities, tape = previous.infer(model, item, session)
            result["free_queries"] += 1
            if (prediction["raw_hex"], prediction["stop_reason"]) != (
                original_rows[item["id"]]["raw_hex"],
                original_rows[item["id"]]["stop_reason"],
            ):
                raise ValueError("original generation differs from frozen final train evaluation")
            row = {
                "id": item["id"],
                "kind": item["kind"],
                "original": prediction,
                "replacements": dict(zip(("target", "distractor"), replacements, strict=True)),
                "read": previous.read_profile(model, item, tape, session, source),
                "original_reconstruction": reconstruct(
                    model, item, session, tape, source, item["answer"]
                ),
            }
            result["teacher_queries"] += 1
            result["reconstruction_queries"] += 1
            for name, variant in changed.items():
                session = f"iterative-read-{name}-{item['id']}"
                prediction, new_probabilities, tape = previous.infer(model, variant, session)
                result["free_queries"] += 1
                row[name] = prediction
                row[f"{name}_first_tv"] = float((new_probabilities - probabilities).abs().sum() / 2)
                index = source if name == "target" else 1 - source
                row[f"{name}_reconstruction"] = reconstruct(
                    model, variant, session, tape, index, row["replacements"][name]
                )
                result["reconstruction_queries"] += 1
            result["rows"].append(row)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        summary = previous.summarize(result["rows"])
        for kind, values in summary.items():
            rows = [row for row in result["rows"] if row["kind"] == kind]
            for variant in ("original", "target", "distractor"):
                profiles = [row[f"{variant}_reconstruction"] for row in rows]
                n = sum(profile["span_tokens"] for profile in profiles)
                values[f"{variant}_span_reconstruction_accuracy"] = (
                    sum(profile["span_correct"] for profile in profiles) / n if n else None
                )
        result.update(
            cpu_seconds=time.process_time(),
            summary=summary,
            parameters_unchanged=before == paired.fingerprint(model.state_dict()),
            checkpoint_unchanged=paired.sha(checkpoint) == CHECKPOINT_SHA,
            manifest_unchanged=paired.sha(calibration.MANIFEST) == calibration.MANIFEST_SHA,
            source_reports_unchanged=paired.sha(fidelity.REPORT) == SOURCE_SHA
            and paired.sha(OLD_REPORT) == OLD_SHA,
            code_unchanged=code_identity() == result["code_identity"],
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 96
            and result["free_queries"] == 288
            and result["teacher_queries"] == 96
            and result["reconstruction_queries"] == 288
            and result["cpu_seconds"] <= 60
            and all(
                result[key]
                for key in (
                    "parameters_unchanged",
                    "checkpoint_unchanged",
                    "manifest_unchanged",
                    "source_reports_unchanged",
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
