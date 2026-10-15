"""PLAN73: report/code-only credit attribution; no models or inference are loaded."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.training import audit_taiji_n1_frozen_writer_exchange as exchange  # noqa: E402

train = exchange.train
paired = train.paired
REPORT = ROOT / "reports/taiji_n1_credit_lineage_20261016.json"
SOURCES = {
    "52": (
        ROOT / "reports/taiji_n1_contextual_question_train_20261015.json",
        "cb26c88e8dab98819a3dbcd48348f7737230872f5444039d460af77e7d00809f",
    ),
    "70": (train.REPORT, exchange.confirmation.TRAIN_SHA),
    "71": (exchange.confirmation.REPORT, exchange.CONFIRMATION_SHA),
    "72": (exchange.REPORT, "a26b9e467f735c813a9b1ec97638bf811d82edfd23f843c95812154c124fa827"),
}
PROTOCOL = dict(
    schema="n1-credit-lineage-v1",
    cpu_limit_seconds=30,
    model_loads=0,
    parameter_updates=0,
    model_queries=0,
    signature_clone_updates=0,
    confirmation_queries=0,
    can_promote=False,
)
ANCHORS = (
    ("scripts/training/_taiji_n1_relation_paired_transfer.py", "migrate"),
    ("scripts/training/run_taiji_n1_relation_paired.py", "expected_step"),
    ("taiji/position_evidence_reference.py", "write"),
    ("taiji/position_evidence_reference.py", "question_state"),
    ("taiji/native_evidence_fidelity.py", "observation_logits"),
    ("taiji/native_paired_question_learning.py", "paired_question_loss"),
)
DOCUMENTS = (
    "plans/active/TAIJI_CORE_REQUIREMENTS.md",
    "plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md",
    "plans/reference/PLAN-N-01_development_revision_20261014.md",
)


def identity():
    return dict(
        protocol=PROTOCOL, runner=paired.sha(Path(__file__)), evidence_code=exchange.identity()
    )


def anchors():
    result = []
    for relative, name in ANCHORS:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        matches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ]
        if len(matches) != 1:
            raise ValueError("unique live implementation anchor required")
        node = matches[0]
        result.append(
            dict(
                file=relative,
                start=node.lineno,
                end=node.end_lineno,
                function=name,
                sha256=paired.sha(path),
            )
        )
    return result


def summarize(records):
    mother, trained, confirmed, swapped = [records[k] for k in ("52", "70", "71", "72")]
    if not all(r["complete"] for r in records.values()):
        raise ValueError("all four complete evidence records required")
    if (
        mother["final_steps_by_parameter_group"]
        != dict(reader=[18432], local=[12288], context=[6144])
        or mother["final_optimizer_state_entries"] != 35
    ):
        raise ValueError("actual52 old moments differ from inheritance contract")
    if mother["protocol"]["final_historical_updates"] != 27648:
        raise ValueError("actual52 historical lineage differs")
    expected = {(seed, arm) for seed in train.SEEDS for arm in train.ARMS}
    if (
        len(trained["models"]) != 9
        or {(m["seed"], m["arm"]) for m in trained["models"]} != expected
    ):
        raise ValueError("nine distinct paired models required")
    writer_steps = []
    for model in trained["models"]:
        writer = {n: v for n, v in model["final_steps"].items() if n.startswith("writer.")}
        target = 18432 if model["arm"] == "cut" else 24576
        if len(writer) != 4 or set(writer.values()) != {target}:
            raise ValueError("phase-specific writer steps differ")
        if model["final_steps"]["embedding.weight"] != 24576:
            raise ValueError("shared embedding must retain other learning paths")
        writer_steps.append(
            dict(seed=model["seed"], arm=model["arm"], writer_steps=writer, embedding_step=24576)
        )
    zero_signatures = [
        c for c in trained["restore_checks"] if Path(c["path"]).name.endswith("_zero.pt")
    ]
    if len(zero_signatures) != 9:
        raise ValueError("nine actual zero independent signatures required")
    for check in zero_signatures:
        signature = check["signature"]
        cut = "_cut_" in Path(check["path"]).name
        if (
            not signature["relation_head_changed"]
            or (cut and (not signature["writer_unchanged"] or signature["writer_gradient_l2"] != 0))
            or (not cut and signature["writer_gradient_l2"] <= 0)
        ):
            raise ValueError("actual engineer credit evidence differs")
    if (
        confirmed["passed"]
        or confirmed["aggregate_exact"] != dict(position=91, terminal=20, cut=92)
        or confirmed["gates"]["advantage_cut"]
    ):
        raise ValueError("preserve original71 failed credit gate")
    changes = []
    for seed in train.SEEDS:
        for arm, condition in (("position", "mother_writer"), ("cut", "position_writer")):
            base = {
                (r["evaluation_set"], r["id"]): r
                for r in swapped["rows"]
                if r["seed"] == seed and r["arm"] == arm and r["condition"] == "native"
            }
            variant = {
                (r["evaluation_set"], r["id"]): r
                for r in swapped["rows"]
                if r["seed"] == seed and r["arm"] == arm and r["condition"] == condition
            }
            if len(base) != 576 or set(base) != set(variant):
                raise ValueError("full same-item writer contrasts required")
            changes.append(
                dict(
                    seed=seed,
                    arm=arm,
                    condition=condition,
                    gained=sum(
                        not base[k]["free_exact"] and variant[k]["free_exact"] for k in base
                    ),
                    lost=sum(base[k]["free_exact"] and not variant[k]["free_exact"] for k in base),
                    raw_changed=sum(base[k]["raw_hex"] != variant[k]["raw_hex"] for k in base),
                )
            )
    return dict(
        mother_writer_adam_step=18432,
        mother_historical_updates=27648,
        independently_reinitialized_parameters=6576,
        total_parameters=62056,
        independent_end_to_end_seeds=False,
        writer_steps=writer_steps,
        writer_exchange_changes=changes,
        credit_scope="only-last6144-update-phase; inherited writer and old moments retained",
        reachable_credit_verified=True,
        training_sufficiency_verified=True,
        unseen_position_exact=91,
        unseen_terminal_exact=20,
        unseen_cut_exact=92,
        original71_passed=False,
        last_phase_writer_net_gain_established=False,
        whole_project_complete=False,
        missing_exits=[
            "original-C2-three-organ-behavior",
            "retention",
            "C4-real-net-gain",
            "same-bundle-three-organ-exits",
            "N2-N6-remaining-exits",
            "HEAD-bound-full-CI",
        ],
        next_action="fresh-full-history-paired-engineering-before-any-new-training",
    )


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive offline audit")
    records = {}
    guards = {}
    for key, (path, expected) in SOURCES.items():
        if paired.sha(path) != expected:
            raise ValueError("frozen evidence bytes changed")
        records[key] = json.loads(path.read_text(encoding="utf-8"))
        guards[str(path)] = expected
    identities = {
        "52": train.runner.transfer.source.code_identity(),
        "70": train.identity(),
        "71": exchange.confirmation.identity(),
        "72": exchange.identity(),
    }
    if any(records[k]["code_identity"] != identities[k] for k in records):
        raise ValueError("live implementation no longer matches frozen reports")
    code_anchors = anchors()
    for entry in code_anchors:
        guards[str(ROOT / entry["file"])] = entry["sha256"]
    context_documents = []
    for relative in DOCUMENTS:
        path = ROOT / relative
        if not path.read_text(encoding="utf-8").strip():
            raise ValueError("root project context required")
        context_documents.append(dict(file=relative, sha256=paired.sha(path)))
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        code_anchors=code_anchors,
        context_documents=context_documents,
        summary=summarize(records),
        model_loads=0,
        model_queries=0,
        parameter_updates=0,
        confirmation_queries=0,
    )
    result["sources_unchanged"] = all(paired.sha(Path(p)) == h for p, h in guards.items())
    result["cpu_seconds"] = time.process_time()
    result["complete"] = result["sources_unchanged"] and result["cpu_seconds"] <= 30
    train.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.execute_approved:
        result = audit()
        print(
            json.dumps(
                dict(
                    complete=result["complete"],
                    cpu_seconds=result["cpu_seconds"],
                    next_action=result["summary"]["next_action"],
                )
            )
        )
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
