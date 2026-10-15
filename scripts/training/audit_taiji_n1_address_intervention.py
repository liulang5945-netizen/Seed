"""Approved PLAN27 oracle address diagnosis; labels never enter model training."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import audit_taiji_n1_fidelity_content_read as single  # noqa: E402
from scripts.training import audit_taiji_n1_iterative_content_read as double  # noqa: E402
from scripts.training import audit_taiji_n1_train_content_read as content  # noqa: E402
from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402
from taiji.native_iterative_evidence_candidate import IterativeEvidenceCandidate  # noqa: E402

REPORT = ROOT / "reports/taiji_n1_address_intervention_20261015.json"
MODES = ("native", "source", "position")
CASES = (
    (
        "single",
        single,
        NativeEvidenceCandidate,
        "taiji_n1_fidelity_content_read_20261015.json",
        "66701478e790b9a540c35ef23d3f66e58fa03dac536a071a2efbe848aaab8649",
    ),
    (
        "double",
        double,
        IterativeEvidenceCandidate,
        "taiji_n1_iterative_content_read_20261015.json",
        "c1bf0a00fbf2ce87c735d331cd7aa58e7e7660659b3e8d54356243606c6aec50",
    ),
)


def addresses(item, tape):
    if item["split"] != "train" or item["kind"] not in ("lookup", "revision"):
        raise ValueError("only material train episodes before any model query")
    answer = item["answer"].encode()
    sources = [
        i
        for i, text in enumerate(item["observations"])
        if text.endswith(f" uses {item['answer']}.")
    ]
    if len(sources) != 1 or len(answer) != 6:
        raise ValueError("unique source and six material bytes required")
    index = sources[0]
    source = f"source-{index + 1}"
    start = item["observations"][index].encode().rfind(answer)
    positions = [tape.positions.index((source, start + i)) for i in range(6)]
    return source, positions


def weights(model, tape, hidden, mode, source, position):
    scores = tape.keys @ model.query(hidden) / model.config.key**0.5
    if mode == "position":
        if tape.positions[position][0] != source:
            raise ValueError("oracle position must belong to oracle source")
        result = torch.zeros_like(scores)
        result[position] = 1
        return result
    if mode == "source":
        mask = torch.tensor([s == source for s, _ in tape.positions], device=scores.device)
        if not mask.any():
            raise ValueError("oracle source absent")
        scores = scores.masked_fill(~mask, float("-inf"))
    elif mode != "native":
        raise ValueError("unknown address intervention")
    return torch.softmax(scores, dim=0)


@torch.no_grad()
def teacher(model, tape, session, question, target, mode, source, positions):
    if tape.session != session or len(positions) != len(target) or mode not in MODES:
        raise ValueError("invalid session, target positions or intervention")
    hidden, previous = model.question_state(question), model.BOUNDARY
    outputs = []
    for index, byte in enumerate((*target, model.BOUNDARY)):
        if index == len(target):
            # Boundary is always native, with the state produced by this condition.
            logits, hidden, _ = model.step(tape, session, hidden, previous)
        else:
            for _ in range(getattr(model, "READ_HOPS", 1)):
                attention = weights(model, tape, hidden, mode, source, positions[index])
                read = attention @ tape.values
                hidden = model.decoder(torch.cat((model.symbol(previous), read)), hidden)
                logits = model.readout(torch.cat((hidden, read)))
        outputs.append(logits)
        previous = byte
    return torch.stack(outputs)


def profile(logits, target):
    labels = torch.tensor([*target, 256], device=logits.device)
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    correct = (logits.argmax(dim=-1) == labels).tolist()
    return {
        "correct": correct,
        "ce": losses.tolist(),
        "teacher_tokens_correct": sum(correct),
        "teacher_ce": float(losses.mean()),
        "six_exact": all(correct[:-1]),
        "seven_exact": all(correct),
    }


def summarize(rows):
    summary = {}
    for candidate, *_ in CASES:
        selected = [row for row in rows if row["candidate"] == candidate]
        modes = {}
        for mode in MODES:
            profiles = [row[mode] for row in selected]
            modes[mode] = {
                "items": len(profiles),
                "six_exact": sum(p["six_exact"] for p in profiles),
                "seven_exact": sum(p["seven_exact"] for p in profiles),
                "byte_correct": [sum(p["correct"][i] for p in profiles) for i in range(7)],
                "mean_ce": [
                    sum(p["ce"][i] for p in profiles) / len(profiles) if profiles else None
                    for i in range(7)
                ],
                "paired_six_gains": sum(
                    row[mode]["six_exact"] and not row["native"]["six_exact"] for row in selected
                ),
                "paired_six_losses": sum(
                    row["native"]["six_exact"] and not row[mode]["six_exact"] for row in selected
                ),
            }
        summary[candidate] = modes
    return summary


def identity():
    return {
        "single": single.code_identity(),
        "double": double.code_identity(),
        "audit": paired.sha(Path(__file__)),
    }


def audit():
    if REPORT.exists():
        raise FileExistsError("address intervention exists; no overwrite or repeat")
    if paired.sha(calibration.MANIFEST) != calibration.MANIFEST_SHA:
        raise ValueError("frozen manifest changed")
    items = content.select(json.loads(calibration.MANIFEST.read_text(encoding="utf-8")))
    result = {
        "schema": "n1-address-intervention-v1",
        "rows": [],
        "teacher_sequences": 0,
        "parameter_updates": 0,
        "free_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "can_promote": False,
        "oracle_is_capability": False,
        "cpu_limit_seconds": 60,
        "manifest_sha256": calibration.MANIFEST_SHA,
        "code_identity": identity(),
        "checks": [],
        "source_hashes": {},
        "status": "running",
    }
    try:
        for name, source, cls, diagnostic_name, diagnostic_sha in CASES:
            diagnostic = ROOT / "reports" / diagnostic_name
            if (
                paired.sha(source.fidelity.REPORT) != source.SOURCE_SHA
                or paired.sha(diagnostic) != diagnostic_sha
            ):
                raise ValueError("frozen source report changed")
            train = json.loads(source.fidelity.REPORT.read_text(encoding="utf-8"))
            reference = json.loads(diagnostic.read_text(encoding="utf-8"))
            if (
                not train["complete"]
                or not reference["complete"]
                or train["protocol"] != source.fidelity.PROTOCOL
            ):
                raise ValueError("complete frozen protocols required")
            final = [r for r in train["restore_checks"] if r["path"].endswith("final.pt")]
            if len(final) != 1 or final[0]["sha256"] != source.CHECKPOINT_SHA:
                raise ValueError("unique frozen final checkpoint required")
            checkpoint = Path(final[0]["path"])
            if paired.sha(checkpoint) != source.CHECKPOINT_SHA:
                raise ValueError("frozen checkpoint changed")
            for path in (source.fidelity.REPORT, diagnostic, checkpoint):
                result["source_hashes"][str(path)] = paired.sha(path)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (
                payload["protocol"] != source.fidelity.PROTOCOL
                or payload["completed_updates"] != 6144
            ):
                raise ValueError("checkpoint protocol changed")
            model = cls.restore_bundle(payload["candidate"])
            before = paired.fingerprint(model.state_dict())
            stored_sources = model.bundle()["sessions"]
            controls = {row["id"]: row["read"] for row in reference["rows"]}
            with torch.no_grad():
                for item in items:
                    if time.process_time() >= 58:
                        raise paired.BudgetStop("address diagnostic CPU reserve reached")
                    session = f"oracle-{name}-{item['id']}"
                    paired.observe(model, paired.project(item), session)
                    tape = model.evidence(session)
                    source_name, positions = addresses(item, tape)
                    target = item["answer"].encode()
                    question = item["question"].encode()
                    row = {"candidate": name, "id": item["id"], "kind": item["kind"]}
                    for mode in MODES:
                        if mode == "native":
                            logits, _ = model.teacher_logits(tape, session, question, target)
                        else:
                            logits = teacher(
                                model, tape, session, question, target, mode, source_name, positions
                            )
                        row[mode] = profile(logits, target)
                        result["teacher_sequences"] += 1
                    control = controls[item["id"]]
                    if (
                        row["native"]["teacher_tokens_correct"] != control["teacher_tokens_correct"]
                        or abs(row["native"]["teacher_ce"] - control["teacher_ce"]) > 1e-6
                    ):
                        raise ValueError("native control differs from frozen diagnostic")
                    # Only ephemeral external-observation sessions were introduced.
                    del model._observations[session]
                    result["rows"].append(row)
            result["checks"].append(
                {
                    "candidate": name,
                    "parameters_unchanged": before == paired.fingerprint(model.state_dict()),
                    "sources_unchanged": model.bundle()["sessions"] == stored_sources,
                    "native_controls_equal": True,
                }
            )
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result.update(
            cpu_seconds=time.process_time(),
            summary=summarize(result["rows"]),
            code_unchanged=identity() == result["code_identity"],
            manifest_unchanged=paired.sha(calibration.MANIFEST) == calibration.MANIFEST_SHA,
            source_files_unchanged=all(
                paired.sha(Path(path)) == sha for path, sha in result["source_hashes"].items()
            ),
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 192
            and result["teacher_sequences"] == 576
            and len(result["checks"]) == 2
            and all(
                all(
                    check[key]
                    for key in (
                        "parameters_unchanged",
                        "sources_unchanged",
                        "native_controls_equal",
                    )
                )
                for check in result["checks"]
            )
            and result["cpu_seconds"] <= 60
            and result["code_unchanged"]
            and result["manifest_unchanged"]
            and result["source_files_unchanged"]
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    torch.set_num_threads(1)
    output = audit()
    print(json.dumps({key: output[key] for key in ("complete", "cpu_seconds", "summary")}))
    raise SystemExit(0 if output["complete"] else 1)
