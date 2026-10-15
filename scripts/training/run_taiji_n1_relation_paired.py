"""PLAN63: nine fixed paired models; confirmation inference is never permitted here."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import _taiji_n1_relation_paired_transfer as transfer  # noqa: E402
from scripts.training import _taiji_n1_v4_seal as seal  # noqa: E402
from scripts.training import audit_taiji_n1_relation_paired_preflight as preflight  # noqa: E402
from scripts.training import run_taiji_n1_local_relation as core  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

paired = core.paired
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
REPORT = ROOT / "reports/taiji_n1_relation_paired_train_20261015.json"
PREFLIGHT_SHA = "868d162122f322f4c75420fa4dd15f9b4e9db38a1581b903328648624beab5ae"
PROTOCOL = {
    **{
        k: v
        for k, v in core.PROTOCOL.items()
        if k not in ("seed", "arm", "updates", "cpu_limit_seconds")
    },
    "schema": "n1-relation-paired-train-v1",
    "seeds": list(SEEDS),
    "arms": list(ARMS),
    "updates_per_model": 6144,
    "total_updates": 55296,
    "total_qa_sequences": 82944,
    "cpu_limit_seconds": 2400,
    "holdout_queries": 0,
    "confirmation_queries": 0,
    "confirmation_original_sha256": seal.SEALS[
        "plans/manifests/n1_relation_confirmation_v4_20261015.json"
    ][0],
}


def identity():
    return dict(
        protocol=PROTOCOL,
        core=core.code_identity(),
        preflight=preflight.identity(),
        helper=paired.sha(Path(transfer.__file__)),
        seal=paired.sha(Path(seal.__file__)),
        runner=paired.sha(Path(__file__)),
    )


def readiness():
    if paired.sha(preflight.REPORT) != PREFLIGHT_SHA:
        raise ValueError("frozen62 report changed")
    ready = json.loads(preflight.REPORT.read_text(encoding="utf-8"))
    if (
        not ready["complete"]
        or ready["code_identity"] != preflight.identity()
        or len(ready["restore_checks"]) != 9
    ):
        raise ValueError("complete unchanged62 with9 recoveries required")
    checked = seal.check_all()
    for name, expected in ready["source_guards"].items():
        rel = Path(name).relative_to(ROOT).as_posix()
        if rel in checked:
            if checked[rel]["original_sha256"] != expected:
                raise ValueError("wrong original seal")
        elif paired.sha(Path(name)) != expected:
            raise ValueError("frozen62 source changed")
    return ready, checked


def fresh(payload, seed, ready):
    model, optimizer = transfer.migrate(payload, seed)
    expected = ready["initial_fingerprints"][str(seed)]
    if len(set(expected)) != 1 or paired.fingerprint(model.bundle(optimizer)) != expected[0]:
        raise ValueError("fresh initial full bundle must match62 pairing")
    return model, optimizer


def save(model, optimizer, path, seed, arm, completed):
    payload = dict(
        protocol=PROTOCOL,
        seed=seed,
        arm=arm,
        completed_updates=completed,
        candidate=model.bundle(optimizer),
        expected_signature=core.signature(model, optimizer),
    )
    with path.open("xb") as f:
        torch.save(payload, f)
    return payload


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["protocol"] != PROTOCOL
        or payload["seed"] not in SEEDS
        or payload["arm"] not in ARMS
    ):
        raise ValueError("frozen paired training identity required")
    model = transfer.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    expected = 35 if payload["completed_updates"] == 0 else 41
    if len(optimizer.state) != expected:
        raise ValueError("full old/new Adam moments required")
    return dict(signature=core.signature(model, optimizer), cpu_seconds=time.process_time())


def independent(path, expected, budget):
    budget.guard(reserve=10)
    try:
        response = subprocess.run(
            [sys.executable, str(Path(__file__)), "--restore-worker", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(30, budget.limit - budget.used() - 2),
            env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            ),
        )
        result = json.loads(response.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += result["cpu_seconds"]
    if result["signature"] != expected:
        raise ValueError("paired independent restore differs")
    budget.guard()
    return dict(path=str(path), sha256=paired.sha(path), equal=True)


def train_step(model, optimizer, item, session, arm, second=None):
    if item["split"] != "train" or arm not in ARMS:
        raise ValueError("fixed train split and arm required")
    if item["kind"] in ("lookup", "unknown") and second is None:
        raise ValueError("paired question required")
    if second is not None and (
        second["split"] != "train"
        or second["parent_id"] != item["id"]
        or second["observations"] != item["observations"]
    ):
        raise ValueError("same train observations required")
    paired.observe(model, paired.project(item), session)
    optimizer.zero_grad(set_to_none=True)
    losses = core.paired_question_loss(
        model,
        session,
        item["question"].encode(),
        item["answer"].encode(),
        second_question=None if second is None else second["question"].encode(),
        second_target=None if second is None else second["answer"].encode(),
        arm=arm,
    )
    if not torch.isfinite(losses.total):
        raise ValueError("nonfinite loss")
    losses.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite gradient")
    optimizer.step()
    return dict(
        total=float(losses.total.detach()),
        qa=float(losses.qa.detach()),
        observation=float(losses.observation.detach()),
        qa_sequences=1 if second is None else 2,
    )


def expected_step(name, arm, completed):
    if arm not in ARMS or completed < 0:
        raise ValueError("frozen arm and nonnegative updates required")
    base = {"reader": 18432, "local": 12288, "context": 6144, "relation": 0}[
        core.parameter_group(name)
    ]
    return base if arm == "cut" and name.startswith("writer.") else base + completed


def sufficient(summary):
    return (
        core.calibration.sufficient(summary)
        and all(
            summary["free_by_kind"].get(k, 0) >= 40
            for k in ("lookup", "revision", "transition", "unknown")
        )
        and summary["pair_exact"] >= 87
        and summary["rename_free_exact"] >= 173
        and all(
            summary["rename_free_by_kind"].get(k, 0) >= 40
            for k in ("lookup", "revision", "transition", "unknown")
        )
        and summary["rename_pair_exact"] >= 87
    )


def evaluate(model, seed, arm, step, views, budget):
    rows = []
    before = paired.fingerprint(model.bundle())
    for view, items in views.items():
        for item in items:
            budget.guard(reserve=10)
            session = f"relation63-{seed}-{arm}-{step}-{view}-{item['id']}"
            try:
                row = core.measure(model, item, arm, session)
                rows.append(
                    dict(row, seed=seed, arm=arm, completed_updates=step, evaluation_set=view)
                )
            finally:
                model._observations.pop(session, None)
    if paired.fingerprint(model.bundle()) != before:
        raise ValueError("evaluation changed model/RNG/history")
    selected = {v: [r for r in rows if r["evaluation_set"] == v] for v in views}
    summary = core.summarize(selected["original"])[str(seed)][arm]
    renamed = core.summarize(selected["rename"])[str(seed)][arm]
    summary.update(
        pair_exact=sum(r["free_exact"] for r in selected["pair"]),
        rename_free_exact=renamed["free_exact"],
        rename_free_by_kind=renamed["free_by_kind"],
        rename_pair_exact=sum(r["free_exact"] for r in selected["rename_pair"]),
    )
    return rows, summary


def audit():
    if REPORT.exists():
        raise FileExistsError("exclusive fixed paired batch, no restart")
    ready, seals = readiness()
    source_record, payload, source_path = transfer.source_payload()
    train = core.checked_train()
    curriculum = core.checked_curriculum(train)
    probes = [
        core.renaming.rename_item(e, p["names"])
        for e, p in zip(train, curriculum["probes"], strict=True)
    ]
    views = dict(
        original=train,
        pair=core.checked_pairs(train),
        rename=probes,
        rename_pair=core.build_pairs(probes),
    )
    budget = paired.CpuBudget(2400)
    folder = Path(get_external_path("data/checkpoints")) / "n1_relation_paired_train_20261015"
    folder.mkdir(exist_ok=False)
    guards = {
        str(p): paired.sha(p)
        for p in (
            preflight.REPORT,
            transfer.source.REPORT,
            source_path,
            core.calibration.MANIFEST,
            core.PAIR_MANIFEST,
            core.CURRICULUM,
            *(ROOT / name for name in seal.SEALS),
        )
    }
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        confirmation_seals=seals,
        completed_updates=0,
        rows=[],
        losses=[],
        models=[],
        restore_checks=[],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        status="running",
        can_promote=False,
    )
    current = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                model, optimizer = fresh(payload, seed, ready)
                current = dict(seed=seed, arm=arm, completed=0)
                path = folder / f"s{seed}_{arm}_zero.pt"
                saved = save(model, optimizer, path, seed, arm, 0)
                result["restore_checks"].append(
                    independent(path, saved["expected_signature"], budget)
                )
                milestones = []
                for step in range(6145):
                    if step in (0, 3072, 6144):
                        rows, summary = evaluate(model, seed, arm, step, views, budget)
                        if (
                            step == 0
                            and arm == "position"
                            and not all(
                                core.transfer_equal(
                                    [r for r in rows if r["evaluation_set"] == v], source_record, v
                                )
                                for v in views
                            )
                        ):
                            raise ValueError("all576 native zero controls differ from52")
                        result["rows"].extend(rows)
                        milestones.append(dict(summary, completed_updates=step))
                        print(
                            f"relation63 seed={seed} arm={arm} step={step} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                            flush=True,
                        )
                    if step == 6144:
                        break
                    budget.guard(reserve=10)
                    item, second = core.renaming.train_item(step, train, curriculum)
                    loss = train_step(
                        model, optimizer, item, f"relation63-train-{seed}-{arm}-{step}", arm, second
                    )
                    result["losses"].append(dict(loss, seed=seed, arm=arm))
                    current["completed"] += 1
                    result["completed_updates"] += 1
                groups = {
                    label: sorted(
                        {
                            int(optimizer.state[p]["step"])
                            for n, p in model.named_parameters()
                            if core.parameter_group(n) == label
                        }
                    )
                    for label in ("reader", "local", "context", "relation")
                }
                expected_groups = {
                    "reader": [18432, 24576] if arm == "cut" else [24576],
                    "local": [18432],
                    "context": [12288],
                    "relation": [6144],
                }
                if groups != expected_groups:
                    raise ValueError("four Adam groups differ from frozen credit policy")
                for name, parameter in model.named_parameters():
                    if int(optimizer.state[parameter]["step"]) != expected_step(name, arm, 6144):
                        raise ValueError("per-parameter Adam step differs from arm policy")
                path = folder / f"s{seed}_{arm}_final.pt"
                saved = save(model, optimizer, path, seed, arm, 6144)
                result["restore_checks"].append(
                    independent(path, saved["expected_signature"], budget)
                )
                result["models"].append(
                    dict(
                        seed=seed,
                        arm=arm,
                        milestones=milestones,
                        training_sufficient=sufficient(milestones[-1]),
                        final_steps=groups,
                    )
                )
                current = None
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no automatic restart")
    finally:
        if current is not None:
            try:
                path = folder / f"s{current['seed']}_{current['arm']}_partial.pt"
                save(model, optimizer, path, current["seed"], current["arm"], current["completed"])
                result["partial_checkpoint"] = dict(
                    path=str(path), sha256=paired.sha(path), **current
                )
            except Exception as error:
                result["partial_error"] = str(error)
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            sources_unchanged=all(paired.sha(Path(p)) == h for p, h in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["models"]) == 9
            and result["completed_updates"] == 55296
            and len(result["rows"]) == 15552
            and sum(x["qa_sequences"] for x in result["losses"]) == 82944
            and len(result["restore_checks"]) == 18
            and result["cpu_seconds"] <= 2400
            and result["sources_unchanged"]
            and result["code_unchanged"]
            and budget.accounting_complete
        )
        positions = [m for m in result["models"] if m["arm"] == "position"]
        result["confirmation_admitted"] = (
            result["complete"]
            and len(positions) == 3
            and all(m["training_sufficient"] for m in positions)
        )
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        print(json.dumps(restore(args.restore_worker)))
    elif args.execute_approved:
        output = audit()
        print(
            json.dumps(
                {
                    k: output.get(k)
                    for k in (
                        "complete",
                        "confirmation_admitted",
                        "completed_updates",
                        "cpu_seconds",
                        "error",
                    )
                }
            )
        )
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
