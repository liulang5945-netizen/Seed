"""PLAN47: frozen activation donors, no training or deployment path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import audit_taiji_n1_cross_read as previous  # noqa: E402
from scripts.training import run_taiji_n1_address_consistency as source  # noqa: E402

paired = source.paired
REPORT = ROOT / "reports/taiji_n1_writer_question_interchange_20261015.json"
SOURCE_SHA = "b19d9986021a8af93bef07e6da15aeb69ad04b7ac77309736ab40a161cbf0f9c"
CONDITIONS = ("native", "writer", "question", "both")


def donor_tape(receiver, donor):
    if receiver.positions != donor.positions or receiver.keys.shape != donor.keys.shape:
        raise ValueError("writer donor must preserve all source positions and dimensions")
    return replace(receiver, states=donor.states, keys=donor.keys, values=donor.values)


@torch.no_grad()
def measure(model, item, session, donor=None, condition="native"):
    if item["split"] != "train":
        raise ValueError("only train-derived inputs allowed")
    if session in model._observations or condition not in CONDITIONS:
        raise ValueError("unique diagnostic session and frozen condition required")
    question_state = model.question_state
    try:
        paired.observe(model, paired.project(item), session)
        tape = model.evidence(session)
        native_tape = tape
        if condition != "native":
            if donor is None or donor["target_symbols"] != [*item["answer"].encode(), 256]:
                raise ValueError("matching original native donor required")
            if condition in ("writer", "both"):
                tape = donor_tape(tape, donor["tape"])
            if condition in ("question", "both"):
                model.question_state = lambda _: donor["question_state"]
        initial = question_state(item["question"].encode())
        logits, addresses = model.teacher_logits(
            tape, session, item["question"].encode(), item["answer"].encode()
        )
        labels = torch.tensor([*item["answer"].encode(), 256])
        predicted = logits.argmax(-1)
        return dict(
            id=item["id"],
            teacher_symbols=predicted.tolist(),
            target_symbols=labels.tolist(),
            teacher_tokens_correct=int((predicted == labels).sum()),
            teacher_exact=bool(torch.equal(predicted, labels)),
            visible_correct=int((predicted[:-1] == labels[:-1]).sum()),
            visible_tokens=len(labels) - 1,
            boundary_correct=bool(predicted[-1] == 256),
            teacher_ce=float(torch.nn.functional.cross_entropy(logits, labels)),
            logits=logits,
            addresses=addresses,
            tape=native_tape,
            question_state=initial,
        )
    finally:
        model.__dict__.pop("question_state", None)
        model._observations.pop(session, None)


def validate_keys(keys):
    expected = {
        (model, view, item["id"], condition)
        for model in ("41", "46")
        for view in ("original", "pair")
        for item in previous.prior.checked_views()[view]
        for condition in ("reference", *CONDITIONS)
    }
    if len(keys) != 2880 or len(set(keys)) != 2880 or set(keys) != expected:
        raise ValueError("all 2880 unique teacher sequences required")


def code_identity():
    return dict(
        source=source.code_identity(),
        audit=paired.sha(Path(__file__)),
        previous=paired.sha(Path(previous.__file__)),
        views=paired.sha(Path(previous.prior.__file__)),
    )


@torch.no_grad()
def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive diagnostic; no automatic repeat")
    if paired.sha(source.REPORT) != SOURCE_SHA:
        raise ValueError("frozen46 report digest mismatch")
    record = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or record["code_identity"] != source.code_identity():
        raise ValueError("complete unchanged46 required")
    _, _, old_path = source.transfer_source()
    (new_entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
    paths = {"41": old_path, "46": Path(new_entry["path"])}
    if paired.sha(paths["46"]) != new_entry["sha256"]:
        raise ValueError("frozen46 checkpoint digest mismatch")
    views = previous.prior.checked_views()
    guards = {
        str(p): paired.sha(p)
        for p in [
            source.REPORT,
            source.pretrained.REPORT,
            source.calibration.MANIFEST,
            source.PAIR_MANIFEST,
            source.CURRICULUM,
            ROOT / "plans/manifests/n1_entity_binding_diagnostic_20261015.json",
            *paths.values(),
        ]
    }
    result = dict(
        schema="n1-writer-question-interchange-v1",
        cpu_limit_seconds=60,
        teacher_queries=0,
        parameter_updates=0,
        free_queries=0,
        dev_queries=0,
        holdout_queries=0,
        rows=[],
        controls=[],
        model_guards=[],
        source_guards=guards,
        code_identity=code_identity(),
        complete=False,
        status="running",
        can_promote=False,
    )
    keys = []
    try:
        for name, path in paths.items():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            expected_protocol = source.pretrained.PROTOCOL if name == "41" else source.PROTOCOL
            if payload["protocol"] != expected_protocol or payload["completed_updates"] != 6144:
                raise ValueError("fixed complete final protocol required")
            model = source.ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(payload["candidate"]["optimizer"])
            before = paired.fingerprint(model.bundle(optimizer))
            if not optimizer.state or before != paired.fingerprint(payload["candidate"]):
                raise ValueError("full checkpoint restore mismatch")
            controls = {
                (r["evaluation_set"], r["id"]): r
                for r in record["rows"]
                if r["completed_updates"] == (0 if name == "41" else 6144)
            }
            for view, renamed_view in (("original", "rename"), ("pair", "rename_pair")):
                for item, renamed in zip(views[view], views[renamed_view], strict=True):
                    reference = None
                    a, b = (controls[v, item["id"]]["free_exact"] for v in (view, renamed_view))
                    combination = (
                        "both" if a and b else "lost" if a else "gained" if b else "neither"
                    )
                    for condition in ("reference", *CONDITIONS):
                        if time.process_time() >= 58:
                            raise paired.BudgetStop("58 second diagnostic reserve")
                        row = measure(
                            model,
                            item if condition == "reference" else renamed,
                            f"writer-question47-{name}-{view}-{item['id']}-{condition}",
                            reference,
                            "native" if condition == "reference" else condition,
                        )
                        result["teacher_queries"] += 1
                        keys.append((name, view, item["id"], condition))
                        if condition in ("reference", "native"):
                            control = controls[
                                view if condition == "reference" else renamed_view, item["id"]
                            ]
                            if (
                                any(
                                    row[k] != control[k]
                                    for k in ("teacher_symbols", "teacher_tokens_correct")
                                )
                                or abs(row["teacher_ce"] - control["teacher_ce"]) > 1e-6
                            ):
                                raise ValueError("native control differs")
                        if condition == "reference":
                            reference = row
                        if condition == "both" and not torch.equal(
                            row["logits"], reference["logits"]
                        ):
                            raise ValueError("both donors must exactly reproduce original logits")
                        if condition in ("reference", "native", "both"):
                            result["controls"].append(
                                dict(
                                    model=name,
                                    view=view,
                                    id=item["id"],
                                    condition=condition,
                                    reproduced=True,
                                )
                            )
                        result["rows"].append(
                            dict(
                                model=name,
                                view=view,
                                id=item["id"],
                                condition=condition,
                                kind=item.get("original_kind", item["kind"]),
                                free_combination=combination,
                                **{
                                    k: v
                                    for k, v in row.items()
                                    if k
                                    not in ("id", "logits", "addresses", "tape", "question_state")
                                },
                                address_tv_visible=float(
                                    (row["addresses"][:-1] - reference["addresses"][:-1])
                                    .abs()
                                    .sum(-1)
                                    .mean()
                                    / 2
                                ),
                                address_js_visible=sum(
                                    previous.prior.js(p, q)
                                    for p, q in zip(
                                        row["addresses"][:-1],
                                        reference["addresses"][:-1],
                                        strict=True,
                                    )
                                )
                                / (len(row["addresses"]) - 1),
                            )
                        )
            unchanged = before == paired.fingerprint(model.bundle(optimizer))
            result["model_guards"].append(dict(model=name, unchanged=unchanged))
            if not unchanged:
                raise ValueError("parameters/Adam/RNG/original history changed")
        validate_keys(keys)
        result.update(complete=True, status="complete", summary=previous.summarize(result["rows"]))
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
        result["code_unchanged"] = result["code_identity"] == code_identity()
        result["cpu_seconds"] = time.process_time()
        result["complete"] = (
            result["complete"]
            and result["sources_unchanged"]
            and result["code_unchanged"]
            and result["cpu_seconds"] <= 60
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.execute_approved:
        output = audit()
        print(
            json.dumps(
                {k: output.get(k) for k in ("complete", "teacher_queries", "cpu_seconds", "error")}
            )
        )
        raise SystemExit(0 if output["complete"] else 1)
    print(json.dumps(dict(read_only=True, teacher_sequences=2880, cpu_limit_seconds=60)))
