"""SPEC-N1-16: train-only frozen-weight fit diagnostic after a failed pilot."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402

BASELINE = ROOT / "reports/taiji_n1_native_evidence_paired_20261015_final.json"
BASELINE_SHA = "ebc6a59623d0805ba8aede2cccc93a1460ed381ac1d860cbb79d78f258a1a9dc"


def training_items(manifest):
    selected = [item for item in manifest["episodes"] if item["split"] == "train"]
    if len(selected) != 96:
        raise ValueError("diagnostic requires all 96 frozen train episodes")
    return selected


@torch.no_grad()
def measure(model, item, arm, session):
    if item["split"] != "train":
        raise ValueError("diagnostic must not query dev or holdout")
    data = paired.project(item)
    paired.observe(model, data, session)
    tape = paired.evidence(model, session, arm)
    target = item["answer"].encode()
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, data.question, target)
    predicted = logits.argmax(dim=-1)
    trace = model.generate_trace(tape, session, data.question, 64)
    return {
        "id": item["id"],
        "kind": item["kind"],
        "target_symbols": labels.tolist(),
        "teacher_symbols": predicted.tolist(),
        "teacher_tokens_correct": int((predicted == labels).sum()),
        "teacher_exact": torch.equal(predicted, labels),
        "teacher_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
        "raw_hex": trace.raw.hex(),
        "output_hex": trace.output.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "free_exact": trace.raw == target and trace.stop_reason == "boundary",
    }


def summarize(rows):
    result = {}
    for seed in paired.SEEDS:
        result[str(seed)] = {}
        for arm in paired.ARMS:
            selected = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
            count = len(selected)
            tokens = sum(len(r["target_symbols"]) for r in selected)
            result[str(seed)][arm] = {
                "items": count,
                "teacher_exact": sum(r["teacher_exact"] for r in selected),
                "free_exact": sum(r["free_exact"] for r in selected),
                "teacher_token_accuracy": (
                    sum(r["teacher_tokens_correct"] for r in selected) / tokens if tokens else None
                ),
                "mean_episode_ce": (
                    sum(r["teacher_ce"] for r in selected) / count if count else None
                ),
                "free_by_kind": {
                    kind: sum(r["free_exact"] for r in selected if r["kind"] == kind)
                    for kind in paired.KINDS
                },
            }
    return result


def audit():
    report = ROOT / "reports/taiji_n1_native_evidence_train_fit_20261015.json"
    if report.exists():
        raise FileExistsError("train-fit diagnostic refuses overwrite or automatic retry")
    if paired.sha(BASELINE) != BASELINE_SHA:
        raise ValueError("formal pilot report digest mismatch")
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    manifest = paired.checked_manifest()
    selected = training_items(manifest)
    checks = [
        entry for entry in baseline["restore_checks"] if entry["checkpoint"].endswith("_trained.pt")
    ]
    if len(checks) != 9:
        raise ValueError("all nine frozen trained checkpoints required")
    result = {
        "schema": "n1-train-fit-diagnostic-v1",
        "baseline_sha256": BASELINE_SHA,
        "rows": [],
        "checkpoint_guards": [],
        "parameter_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "cpu_limit_seconds": 60,
        "code_identity": paired.identity(),
        "audit_code_sha256": paired.sha(Path(__file__)),
    }
    status = "running"
    try:
        for check in checks:
            path = Path(check["checkpoint"])
            if paired.sha(path) != check["sha256"]:
                raise ValueError("frozen trained checkpoint changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            model = NativeEvidenceCandidate.restore_bundle(payload["candidate"])
            before = paired.fingerprint(model.state_dict())
            for item in selected:
                if time.process_time() >= 59:
                    raise paired.BudgetStop("train-fit CPU reserve reached")
                row = measure(model, item, payload["arm"], f"audit-{item['id']}")
                result["rows"].append(dict(row, seed=payload["seed"], arm=payload["arm"]))
            unchanged = (
                before == paired.fingerprint(model.state_dict())
                and paired.sha(path) == check["sha256"]
            )
            result["checkpoint_guards"].append(
                {"seed": payload["seed"], "arm": payload["arm"], "unchanged": unchanged}
            )
            if not unchanged:
                raise ValueError("train-fit diagnostic changed parameters or source checkpoint")
        status = "complete"
    except paired.BudgetStop as error:
        status = "budget-stop"
        result["error"] = str(error)
    except Exception as error:
        status = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result.update(
            status=status, cpu_seconds=time.process_time(), summary=summarize(result["rows"])
        )
        result["complete"] = (
            status == "complete"
            and len(result["rows"]) == 864
            and len(result["checkpoint_guards"]) == 9
            and result["cpu_seconds"] <= 60
        )
        paired.write_new(report, result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = audit()
    print(
        json.dumps({key: output[key] for key in ("status", "complete", "cpu_seconds", "summary")})
    )
    raise SystemExit(0 if output["complete"] else 1)
