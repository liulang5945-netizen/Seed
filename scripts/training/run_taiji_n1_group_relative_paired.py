"""PLAN70: fixed nine-model bounded-Adam training; no confirmation inference."""

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

from scripts.training import audit_taiji_n1_group_relative_paired as engineering  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

runner = engineering.runner
paired = runner.paired
update = engineering.update
SEEDS = runner.SEEDS
ARMS = runner.ARMS
REPORT = ROOT / "reports/taiji_n1_group_relative_paired_train_20261015.json"
PREFLIGHT_SHA = "938bee47ae766c24efdfc20c1e43c49332e518e4d7e59e65c78ba1c57dd743c7"
PROTOCOL = {
    **runner.PROTOCOL,
    "schema": "n1-group-relative-paired-train-v1",
    "update_policy": update.POLICY,
    "preflight_sha256": PREFLIGHT_SHA,
    "signature_clone_updates": 36,
    "signature_clone_qa_sequences": 72,
    "initialization": "fresh52-full-Adam-migration-matches62-and69",
    "report_encoding": "lossless-compact-json",
}


def identity():
    return dict(
        protocol=PROTOCOL, engineering=engineering.identity(), runner=paired.sha(Path(__file__))
    )


def readiness():
    if paired.sha(engineering.REPORT) != PREFLIGHT_SHA:
        raise ValueError("frozen69 engineering report required")
    evidence = json.loads(engineering.REPORT.read_text(encoding="utf-8"))
    if (
        not evidence["complete"]
        or evidence["code_identity"] != engineering.identity()
        or len(evidence["restore_checks"]) != 9
        or evidence["signature_clone_updates"] != 18
        or evidence["main_trajectory_updates"] != 0
        or evidence["confirmation_queries"] != 0
    ):
        raise ValueError("complete unchanged69 zero-update engineering required")
    ready, guards, seals = engineering.readiness()
    if evidence["initial_fingerprints"] != ready["initial_fingerprints"]:
        raise ValueError("69 and62 complete initial pairing differs")
    guards[str(engineering.REPORT)] = PREFLIGHT_SHA
    for check in evidence["restore_checks"]:
        if not check["equal"] or paired.sha(Path(check["path"])) != check["sha256"]:
            raise ValueError("69 independent checkpoint bytes changed")
        guards[check["path"]] = check["sha256"]
    return ready, guards, seals


def load(payload):
    return engineering.load(payload, protocol=PROTOCOL, allowed=(0, 6144))


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=engineering.signature(
            model, optimizer, payload["seed"], payload["arm"], payload["completed_updates"]
        ),
        cpu_seconds=time.process_time(),
        signature_clone_updates=1,
    )


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
    budget.guard()
    if result["signature"] != expected or result["signature_clone_updates"] != 1:
        raise ValueError("independent fixed-training same-arm recovery differs")
    return dict(path=str(path), sha256=paired.sha(path), equal=True, **result)


def save(model, optimizer, path, seed, arm, completed):
    return engineering.save(model, optimizer, path, seed, arm, completed, protocol=PROTOCOL)


def check_final(model, optimizer, source_sessions, arm):
    if len(optimizer.state) != 41:
        raise ValueError("all41 final Adam states required")
    for name, parameter in model.named_parameters():
        if int(optimizer.state[parameter]["step"]) != runner.expected_step(name, arm, 6144):
            raise ValueError("fixed final per-parameter arm Adam steps differ")
    sessions = model.bundle()["sessions"]
    unchanged = all(sessions.get(k) == v for k, v in source_sessions.items())
    count = len(sessions) - len(source_sessions)
    if not unchanged or count != 6144:
        raise ValueError("source sessions changed or fixed history count differs")
    return dict(source_sessions_unchanged=unchanged, new_session_count=count)


def write_new(path, result):
    """Exclusive lossless JSON; retain all rows and measurements without indentation."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def audit():
    folder = Path(get_external_path("data/checkpoints")) / "n1_group_relative_paired_train_20261015"
    if REPORT.exists() or folder.exists():
        raise FileExistsError("exclusive fixed nine-model batch, no restart or overwrite")
    budget = paired.CpuBudget(PROTOCOL["cpu_limit_seconds"])
    ready, guards, seals = readiness()
    _, source, source_path = runner.transfer.source_payload()
    guards[str(source_path)] = paired.sha(source_path)
    baseline = json.loads(runner.REPORT.read_text(encoding="utf-8"))
    train = runner.core.checked_train()
    curriculum = runner.core.checked_curriculum(train)
    probes = [
        runner.core.renaming.rename_item(e, p["names"])
        for e, p in zip(train, curriculum["probes"], strict=True)
    ]
    views = dict(
        original=train,
        pair=runner.core.checked_pairs(train),
        rename=probes,
        rename_pair=runner.core.build_pairs(probes),
    )
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        confirmation_seals=seals,
        initial_fingerprints={str(seed): [] for seed in SEEDS},
        completed_updates=0,
        signature_clone_updates=0,
        losses=[],
        measurements=[],
        rows=[],
        models=[],
        restore_checks=[],
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    current = None
    model = optimizer = None
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=10)
                model, optimizer = runner.fresh(source, seed, ready)
                current = dict(seed=seed, arm=arm, completed=0)
                result["initial_fingerprints"][str(seed)].append(
                    paired.fingerprint(model.bundle(optimizer))
                )
                path = folder / f"s{seed}_{arm}_zero.pt"
                saved = save(model, optimizer, path, seed, arm, 0)
                result["signature_clone_updates"] += 1
                result["restore_checks"].append(
                    independent(path, saved["expected_signature"], budget)
                )
                result["signature_clone_updates"] += 1
                milestones = []
                for step in range(6145):
                    if step in (0, 3072, 6144):
                        rows, summary = runner.evaluate(model, seed, arm, step, views, budget)
                        if step == 0 and arm == "position":
                            expected = [
                                r
                                for r in baseline["rows"]
                                if r["seed"] == seed
                                and r["arm"] == arm
                                and r["completed_updates"] == 0
                            ]
                            if len(expected) != 576 or rows != expected:
                                raise ValueError(
                                    "full576 zero rows must reproduce original63 same seed"
                                )
                        result["rows"].extend(rows)
                        milestones.append(dict(summary, completed_updates=step))
                        print(
                            f"bounded70 seed={seed} arm={arm} step={step} original={summary['free_exact']} pair={summary['pair_exact']} rename={summary['rename_free_exact']} rename_pair={summary['rename_pair_exact']} cpu={budget.used():.2f}",
                            flush=True,
                        )
                    if step == 6144:
                        break
                    budget.guard(reserve=10)
                    item, second = runner.core.renaming.train_item(step, train, curriculum)
                    loss, measurements = update.train_step(
                        model, optimizer, item, f"bounded70-train-{seed}-{arm}-{step}", arm, second
                    )
                    current["completed"] += 1
                    result["completed_updates"] += 1
                    result["losses"].append(dict(loss, seed=seed, arm=arm, step=step))
                    result["measurements"].append(
                        dict(seed=seed, arm=arm, step=step, groups=measurements)
                    )
                history = check_final(model, optimizer, source["candidate"]["sessions"], arm)
                path = folder / f"s{seed}_{arm}_final.pt"
                saved = save(model, optimizer, path, seed, arm, 6144)
                result["signature_clone_updates"] += 1
                result["restore_checks"].append(
                    independent(path, saved["expected_signature"], budget)
                )
                result["signature_clone_updates"] += 1
                result["models"].append(
                    dict(
                        seed=seed,
                        arm=arm,
                        milestones=milestones,
                        training_sufficient=runner.sufficient(milestones[-1]),
                        final_steps={
                            n: int(optimizer.state[p]["step"]) for n, p in model.named_parameters()
                        },
                        **history,
                    )
                )
                current = None
        if result["initial_fingerprints"] != ready["initial_fingerprints"]:
            raise ValueError("full initial pairing differs from62/69")
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no automatic restart")
    finally:
        if current is not None and model is not None and optimizer is not None:
            try:
                path = folder / f"s{current['seed']}_{current['arm']}_partial.pt"
                save(model, optimizer, path, current["seed"], current["arm"], current["completed"])
                result["signature_clone_updates"] += 1
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
        result["signature_clone_qa_sequences"] = 2 * result["signature_clone_updates"]
        result["complete"] = (
            result["status"] == "complete"
            and len(result["models"]) == 9
            and result["completed_updates"] == 55296
            and len(result["rows"]) == 15552
            and len(result["measurements"]) == 55296
            and sum(x["qa_sequences"] for x in result["losses"]) == 82944
            and result["signature_clone_updates"] == 36
            and len(result["restore_checks"]) == 18
            and result["cpu_seconds"] <= 2400
            and all(
                result[k]
                for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
            )
        )
        positions = [m for m in result["models"] if m["arm"] == "position"]
        result["confirmation_admitted"] = (
            result["complete"]
            and len(positions) == 3
            and all(m["training_sufficient"] for m in positions)
        )
        write_new(REPORT, result)
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
