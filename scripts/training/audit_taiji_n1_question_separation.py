"""PLAN56: frozen binding geometry only, never teacher or free generation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from scripts.training import audit_taiji_n1_read_equivariance as views_source  # noqa: E402
from scripts.training import run_taiji_n1_bound_consistency as source  # noqa: E402

paired = source.paired
REPORT = ROOT / "reports/taiji_n1_question_separation_20261015.json"
SOURCE_SHA = "321b025089eea055bdd00447376d3aabf263599be911702e50192da71e04c446"
VIEWS = ("original", "pair", "rename", "rename_pair")


def distance(a, b):
    if a.shape != b.shape or not a.numel():
        raise ValueError("aligned nonempty states required")
    return float((1 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)).mean())


def ratio(a, b):
    return a / b if b > 1e-8 else None


@torch.no_grad()
def capture(model, item, session):
    if item["split"] != "train" or session in model._observations:
        raise ValueError("train and unique session required")
    try:
        paired.observe(model, paired.project(item), session)
        tape = model.evidence(session)
        state, _, _ = model.bind_question(tape, model.question_state(item["question"].encode()))
        return dict(
            hidden=state.hidden,
            states=state.question_states,
            positions=tape.positions,
            hidden_norm=float(state.hidden.norm()),
            states_norm=float(state.question_states.norm(dim=-1).mean()),
        )
    finally:
        model._observations.pop(session, None)


def validate_keys(keys, parents):
    expected = {
        (name, parent, view) for name in ("52", "55") for parent in parents for view in VIEWS
    }
    if len(keys) != 768 or len(set(keys)) != 768 or set(keys) != expected:
        raise ValueError("all768 unique bindings required")


def identity():
    return dict(
        source=source.code_identity(),
        audit=paired.sha(Path(__file__)),
        views=paired.sha(Path(views_source.__file__)),
    )


def summarize(rows):
    summaries = []
    for name in ("52", "55"):
        for kind in ("all", "lookup", "unknown"):
            for combination in ("all", "both", "lost", "gained", "neither"):
                selected = [
                    r
                    for r in rows
                    if r["model"] == name
                    and (kind == "all" or r["kind"] == kind)
                    and (combination == "all" or r["rename_pair_combination"] == combination)
                ]
                if selected:
                    summaries.append(
                        dict(
                            model=name,
                            kind=kind,
                            combination=combination,
                            items=len(selected),
                            means={
                                key: sum(r["metrics"][key] for r in selected) / len(selected)
                                for key in selected[0]["metrics"]
                                if all(r["metrics"][key] is not None for r in selected)
                            },
                        )
                    )
    return summaries


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive diagnostic, no repeat")
    if paired.sha(source.REPORT) != SOURCE_SHA:
        raise ValueError("frozen55 report mismatch")
    records = {
        "52": source.transfer_source()[0],
        "55": json.loads(source.REPORT.read_text(encoding="utf-8")),
    }
    if not records["55"]["complete"] or records["55"]["code_identity"] != source.code_identity():
        raise ValueError("complete unchanged55 required")
    entries = {
        name: next(e for e in r["restore_checks"] if Path(e["path"]).name == "final.pt")
        for name, r in records.items()
    }
    views = views_source.checked_views()
    maps = {v: {r.get("parent_id", r["id"]): r for r in views[v]} for v in VIEWS}
    parents = sorted(maps["pair"])
    if len(parents) != 96:
        raise ValueError("96 fixed train parents required")
    paths = [
        source.REPORT,
        source.prior.REPORT,
        source.calibration.MANIFEST,
        source.PAIR_MANIFEST,
        source.CURRICULUM,
        ROOT / "plans/manifests/n1_entity_binding_diagnostic_20261015.json",
    ]
    paths.extend(Path(e["path"]) for e in entries.values())
    guards = {str(p): paired.sha(p) for p in paths}
    result = dict(
        schema="n1-question-separation-v1",
        bindings=0,
        parameter_updates=0,
        teacher_queries=0,
        free_queries=0,
        dev_queries=0,
        holdout_queries=0,
        source_guards=guards,
        code_identity=identity(),
        rows=[],
        model_guards=[],
        complete=False,
        status="running",
        can_promote=False,
    )
    keys = []
    controls = {
        name: {
            (r["evaluation_set"], r["id"]): r
            for r in record["rows"]
            if r["completed_updates"] == 6144
        }
        for name, record in records.items()
    }
    try:
        for name, entry in entries.items():
            path = Path(entry["path"])
            if paired.sha(path) != entry["sha256"]:
                raise ValueError("source checkpoint changed")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            expected = source.prior.PROTOCOL if name == "52" else source.PROTOCOL
            if payload["protocol"] != expected or payload["completed_updates"] != 6144:
                raise ValueError("fixed source protocol required")
            model = source.ContextualQuestionEvidenceCandidate.restore_bundle(payload["candidate"])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            optimizer.load_state_dict(payload["candidate"]["optimizer"])
            before = paired.fingerprint(model.bundle(optimizer))
            if before != paired.fingerprint(payload["candidate"]) or len(optimizer.state) != len(
                list(model.parameters())
            ):
                raise ValueError("full model/Adam restore required")
            for parent in parents:
                captured = {}
                for view in VIEWS:
                    if time.process_time() >= 58:
                        raise paired.BudgetStop("58 second reserve")
                    item = maps[view][parent]
                    counterpart = maps[
                        (
                            "original"
                            if view == "rename"
                            else "pair" if view == "rename_pair" else view
                        )
                    ][parent]
                    if item["answer"] != counterpart["answer"]:
                        raise ValueError("renaming target mismatch")
                    captured[view] = capture(model, item, f"separation56-{name}-{parent}-{view}")
                    keys.append((name, parent, view))
                    result["bindings"] += 1
                if len({tuple(c["positions"]) for c in captured.values()}) != 1:
                    raise ValueError("source positions must align")
                metrics = {}
                for field in ("hidden", "states"):
                    for label, a, b in (
                        ("invariance", "original", "rename"),
                        ("pair_invariance", "pair", "rename_pair"),
                        ("separation", "original", "pair"),
                        ("rename_separation", "rename", "rename_pair"),
                    ):
                        metrics[field + "_" + label] = distance(
                            captured[a][field], captured[b][field]
                        )
                    metrics[field + "_invariance_over_separation"] = ratio(
                        metrics[field + "_invariance"], metrics[field + "_separation"]
                    )
                    metrics[field + "_pair_invariance_over_rename_separation"] = ratio(
                        metrics[field + "_pair_invariance"], metrics[field + "_rename_separation"]
                    )
                for view, c in captured.items():
                    for field in ("hidden_norm", "states_norm"):
                        metrics[view + "_" + field] = c[field]
                combination = {}
                for view in VIEWS:
                    item = maps[view][parent]
                    a, b = (controls[n][view, item["id"]]["free_exact"] for n in ("52", "55"))
                    combination[view] = (
                        "both" if a and b else "lost" if a else "gained" if b else "neither"
                    )
                result["rows"].append(
                    dict(
                        model=name,
                        parent_id=parent,
                        kind=maps["original"][parent]["kind"],
                        rename_pair_combination=combination["rename_pair"],
                        free_combinations=combination,
                        metrics=metrics,
                    )
                )
            unchanged = before == paired.fingerprint(model.bundle(optimizer))
            result["model_guards"].append(dict(model=name, unchanged=unchanged))
            if not unchanged:
                raise ValueError("model/Adam/RNG/history changed")
        validate_keys(keys, parents)
        result.update(complete=True, status="complete", summary=summarize(result["rows"]))
    except Exception as error:
        result.update(status="rejected", error=f"{type(error).__name__}: {error}")
    finally:
        result["cpu_seconds"] = time.process_time()
        result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
        result["code_unchanged"] = identity() == result["code_identity"]
        result["complete"] &= (
            result["sources_unchanged"] and result["code_unchanged"] and result["cpu_seconds"] <= 60
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
            json.dumps({k: output.get(k) for k in ("complete", "bindings", "cpu_seconds", "error")})
        )
        raise SystemExit(0 if output["complete"] else 1)
    print(json.dumps(dict(read_only=True, bindings=768, cpu_limit_seconds=60)))
