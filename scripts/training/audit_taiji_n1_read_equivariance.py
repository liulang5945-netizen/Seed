"""PLAN44: frozen train-derived teacher-only read trajectory comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import run_taiji_n1_renaming as source  # noqa: E402

paired = source.paired
REPORT = ROOT / "reports/taiji_n1_read_equivariance_20261015.json"
SOURCE_SHA = "92abbff29422a5ba6fec2044f8ebba0f6f833d15a699e3088ef5a0df7ab70d8f"


def js(p, q):
    """Natural-log Jensen-Shannon divergence, with exact zero support allowed."""
    middle = (p + q) / 2
    tiny = torch.finfo(p.dtype).tiny
    return float(
        (
            (p * (p.clamp_min(tiny).log() - middle.clamp_min(tiny).log())).sum()
            + (q * (q.clamp_min(tiny).log() - middle.clamp_min(tiny).log())).sum()
        )
        / 2
    )


def relative_l2(a, b):
    return float((a - b).norm() / (a.norm() + b.norm() + 1e-12))


def checked_views():
    train = source.checked_train()
    curriculum = source.checked_curriculum(train)
    renamed = [
        source.renaming.rename_item(item, entry["names"])
        for item, entry in zip(train, curriculum["probes"], strict=True)
    ]
    return {
        "original": train,
        "pair": source.checked_pairs(train),
        "rename": renamed,
        "rename_pair": source.build_pairs(renamed),
    }


def validate_denominator(keys):
    expected = {
        (model, view, item["id"])
        for model in ("41", "43")
        for view, items in checked_views().items()
        for item in items
    }
    if len(keys) != 1152 or len(set(keys)) != 1152 or set(keys) != expected:
        raise ValueError("all 1152 unique fixed teacher sequences required")


@torch.no_grad()
def capture(model, item, session):
    if item["split"] != "train":
        raise ValueError("only train-derived inputs allowed")
    if session in model._observations:
        raise ValueError("diagnostic session collides with frozen history")
    original_step = model.step
    steps = []

    def traced(*args, **kwargs):
        logits, state, weights = original_step(*args, **kwargs)
        steps.append(
            {
                "address": weights.detach().clone(),
                "read": (weights @ state.conditioned_tape.values).detach().clone(),
                "hidden": state.hidden.detach().clone(),
                "logits": logits.detach().clone(),
                "alignment": state.alignment.detach(),
            }
        )
        return logits, state, weights

    try:
        paired.observe(model, paired.project(item), session)
        tape = model.evidence(session)
        model.step = traced
        target = item["answer"].encode()
        logits, _ = model.teacher_logits(tape, session, item["question"].encode(), target)
        labels = torch.tensor([*target, 256])
        predicted = logits.argmax(-1)
        if len(steps) != len(labels):
            raise ValueError("one captured second-hop trajectory per emitted symbol required")
        return dict(
            id=item["id"],
            kind=item["kind"],
            target_symbols=labels.tolist(),
            teacher_symbols=predicted.tolist(),
            teacher_tokens_correct=int((predicted == labels).sum()),
            teacher_ce=float(torch.nn.functional.cross_entropy(logits, labels)),
            positions=tape.positions,
            steps=steps,
        )
    finally:
        # Restore instance lookup as well as callable semantics.
        model.__dict__.pop("step", None)
        model._observations.pop(session, None)


def compare(left, right):
    if left["positions"] != right["positions"] or left["target_symbols"] != right["target_symbols"]:
        raise ValueError("renamed sources must preserve positions and output targets")
    rows = []
    for index, (a, b) in enumerate(zip(left["steps"], right["steps"], strict=True)):
        rows.append(
            dict(
                index=index,
                boundary=index == len(left["steps"]) - 1,
                address_js=js(a["address"], b["address"]),
                address_tv=float((a["address"] - b["address"]).abs().sum() / 2),
                read_relative_l2=relative_l2(a["read"], b["read"]),
                hidden_relative_l2=relative_l2(a["hidden"], b["hidden"]),
                logit_js=js(a["logits"].softmax(-1), b["logits"].softmax(-1)),
                original_correct=left["teacher_symbols"][index] == left["target_symbols"][index],
                renamed_correct=right["teacher_symbols"][index] == right["target_symbols"][index],
            )
        )
    return rows


def alignment_density(trace, item):
    question = item["question"]
    entities = [n for n in item["entities"] if n in question]
    if len(entities) != 1:
        raise ValueError("one question entity required for offline alignment summary")
    start = question.index(entities[0])
    mask = torch.zeros(len(question), dtype=torch.bool)
    mask[start : start + len(entities[0])] = True
    mass = trace["steps"][0]["alignment"].mean(0)
    return {
        "entity_density": float(mass[mask].mean()),
        "template_density": float(mass[~mask].mean()),
        "entity_mass": float(mass[mask].sum()),
    }


def summaries(rows):
    result = []
    for model in ("41", "43"):
        for view in ("original", "pair"):
            for kind in paired.KINDS:
                selected = [
                    r for r in rows if (r["model"], r["view"], r["kind"]) == (model, view, kind)
                ]
                if not selected:
                    continue
                for combination in ("all", "both", "lost", "gained", "neither"):
                    group = (
                        selected
                        if combination == "all"
                        else [r for r in selected if r["free_combination"] == combination]
                    )
                    if not group:
                        continue
                    for boundary in (False, True):
                        values = [b for r in group for b in r["bytes"] if b["boundary"] == boundary]
                        result.append(
                            dict(
                                model=model,
                                view=view,
                                kind=kind,
                                combination=combination,
                                episodes=len(group),
                                boundary=boundary,
                                bytes=len(values),
                                **{
                                    key: sum(b[key] for b in values) / len(values)
                                    for key in (
                                        "address_js",
                                        "address_tv",
                                        "read_relative_l2",
                                        "hidden_relative_l2",
                                        "logit_js",
                                        "original_correct",
                                        "renamed_correct",
                                    )
                                },
                            )
                        )
    return result


@torch.no_grad()
def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive diagnostic; no automatic repeat")
    if paired.sha(source.REPORT) != SOURCE_SHA:
        raise ValueError("frozen 43 report changed")
    record = json.loads(source.REPORT.read_text(encoding="utf-8"))
    if not record["complete"] or record["code_identity"] != source.code_identity():
        raise ValueError("complete frozen source code required")
    _, _, old_path = source.transfer_source()
    (new_entry,) = [e for e in record["restore_checks"] if Path(e["path"]).name == "final.pt"]
    paths = {"41": old_path, "43": Path(new_entry["path"])}
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
    code = {"source": source.code_identity(), "audit_sha256": paired.sha(Path(__file__))}
    result = dict(
        schema="n1-read-equivariance-v1",
        cpu_limit_seconds=60,
        teacher_queries=0,
        free_queries=0,
        parameter_updates=0,
        dev_queries=0,
        holdout_queries=0,
        rows=[],
        controls=[],
        model_guards=[],
        source_guards=guards,
        code_identity=code,
        can_promote=False,
        status="running",
        complete=False,
    )
    keys = []
    try:
        views = checked_views()
        for name, path in paths.items():
            if name == "43" and paired.sha(path) != new_entry["sha256"]:
                raise ValueError("43 final checkpoint digest mismatch")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            expected_protocol = source.pretrained.PROTOCOL if name == "41" else source.PROTOCOL
            if payload["protocol"] != expected_protocol or payload["completed_updates"] != 6144:
                raise ValueError("fixed final checkpoint protocol required")
            model = source.ResidualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(payload["candidate"]["optimizer"])
            if not optimizer.state:
                raise ValueError("full trained Adam required")
            before = paired.fingerprint(model.bundle(optimizer))
            if before != paired.fingerprint(payload["candidate"]):
                raise ValueError("complete parameter/Adam/RNG/session restore mismatch")
            controls = {
                (r["evaluation_set"], r["id"]): r
                for r in record["rows"]
                if r["completed_updates"] == (0 if name == "41" else 6144)
            }
            for view, renamed_view in (("original", "rename"), ("pair", "rename_pair")):
                for item, renamed in zip(views[view], views[renamed_view], strict=True):
                    traces = []
                    for subset, entry in ((view, item), (renamed_view, renamed)):
                        if time.process_time() >= 58:
                            raise paired.BudgetStop("58 second diagnostic reserve")
                        trace = capture(
                            model, entry, f"read-equivariance44-{name}-{subset}-{entry['id']}"
                        )
                        result["teacher_queries"] += 1
                        keys.append((name, subset, entry["id"]))
                        old = controls[subset, entry["id"]]
                        if (
                            any(
                                trace[k] != old[k]
                                for k in (
                                    "target_symbols",
                                    "teacher_symbols",
                                    "teacher_tokens_correct",
                                )
                            )
                            or abs(trace["teacher_ce"] - old["teacher_ce"]) > 1e-6
                        ):
                            raise ValueError("teacher control mismatch")
                        result["controls"].append(
                            dict(
                                model=name,
                                view=subset,
                                id=entry["id"],
                                teacher_symbols=trace["teacher_symbols"],
                                teacher_ce=trace["teacher_ce"],
                                teacher_tokens_correct=trace["teacher_tokens_correct"],
                                reproduced=True,
                                alignment=alignment_density(trace, entry),
                            )
                        )
                        traces.append(trace)
                    a, b = (controls[v, item["id"]]["free_exact"] for v in (view, renamed_view))
                    result["rows"].append(
                        dict(
                            model=name,
                            view=view,
                            id=item["id"],
                            kind=item.get("original_kind", item["kind"]),
                            free_combination=(
                                "both" if a and b else "lost" if a else "gained" if b else "neither"
                            ),
                            original_free_exact=a,
                            renamed_free_exact=b,
                            bytes=compare(*traces),
                        )
                    )
            unchanged = before == paired.fingerprint(model.bundle(optimizer))
            result["model_guards"].append(dict(model=name, unchanged=unchanged))
            if not unchanged:
                raise ValueError("model/Adam/RNG/history changed")
        validate_denominator(keys)
        result.update(complete=True, status="complete", summary=summaries(result["rows"]))
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["sources_unchanged"] = all(
            paired.sha(Path(p)) == digest for p, digest in guards.items()
        )
        result["code_unchanged"] = code == {
            "source": source.code_identity(),
            "audit_sha256": paired.sha(Path(__file__)),
        }
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
    print(json.dumps({"read_only": True, "teacher_sequences": 1152, "cpu_limit_seconds": 60}))
