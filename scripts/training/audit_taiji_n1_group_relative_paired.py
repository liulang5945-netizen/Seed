"""PLAN69: nine paired zero bundles and independent same-arm bounded-update recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from scripts.training import _taiji_n1_group_relative_paired_adam as update  # noqa: E402
from scripts.training import run_taiji_n1_group_relative_adam as calibration  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402

runner = update.runner
paired = runner.paired
SEEDS = runner.SEEDS
ARMS = runner.ARMS
REPORT = ROOT / "reports/taiji_n1_group_relative_paired_preflight_20261015.json"
CALIBRATION_SHA = "9484a7abd47bde2d61fcc084af7f64a51e39a95392f006222345fbfec6e30248"
PROTOCOL = dict(
    schema="n1-group-relative-paired-preflight-v1",
    seeds=list(SEEDS),
    arms=list(ARMS),
    update_policy=update.POLICY,
    calibration_sha256=CALIBRATION_SHA,
    main_trajectory_updates=0,
    signature_clone_updates=18,
    synthetic_qa_sequences=36,
    cpu_limit_seconds=180,
    curriculum_queries=0,
    dev_queries=0,
    holdout_queries=0,
    confirmation_queries=0,
)
SYNTHETIC = calibration.engineering.SYNTHETIC


def identity():
    return dict(
        protocol=PROTOCOL,
        calibration=calibration.identity(),
        wrapper=paired.sha(Path(update.__file__)),
        runner=paired.sha(Path(__file__)),
    )


def signature(model, optimizer, seed, arm, completed=0):
    if seed not in SEEDS or arm not in ARMS:
        raise ValueError("frozen seed and arm required")
    before = paired.fingerprint(model.bundle(optimizer))
    with torch.random.fork_rng():
        candidate = deepcopy(model.bundle(optimizer))
        probe = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(candidate)
        opt = torch.optim.Adam(probe.parameters(), lr=0.001)
        opt.load_state_dict(candidate["optimizer"])
        writer = {
            n: (p.detach().clone(), deepcopy(opt.state[p]))
            for n, p in probe.named_parameters()
            if n.startswith("writer.")
        }
        head = probe.relation_projection.weight.detach().clone()
        loss, measurements = update.train_step(
            probe,
            opt,
            SYNTHETIC,
            f"bounded69-signature-{seed}-{arm}",
            arm,
            runner.core.renaming.second_question(SYNTHETIC),
            audit=True,
        )
        if len(opt.state) != 41:
            raise ValueError("all41 old/new moments required after mechanical step")
        for name, parameter in probe.named_parameters():
            if int(opt.state[parameter]["step"]) != runner.expected_step(name, arm, completed + 1):
                raise ValueError("per-parameter control Adam step differs")
        writer_unchanged = all(
            torch.equal(p, writer[n][0])
            and paired.fingerprint(opt.state[p]) == paired.fingerprint(writer[n][1])
            for n, p in probe.named_parameters()
            if n in writer
        )
        writer_gradient = measurements["writer"]["gradient_l2"]
        if arm == "cut" and (not writer_unchanged or writer_gradient != 0):
            raise ValueError("cut writer credit, parameters or moments changed")
        if completed == 0 and arm != "cut" and writer_gradient <= 0:
            raise ValueError("position/terminal native writer credit must remain")
        head_changed = not torch.equal(head, probe.relation_projection.weight)
        if completed == 0 and not head_changed:
            raise ValueError("new zero relation head must learn in every arm")
        result = dict(
            bundle=before,
            next_bundle=paired.fingerprint(probe.bundle(opt)),
            next_loss=loss,
            next_measurements=measurements,
            writer_unchanged=writer_unchanged,
            writer_gradient_l2=writer_gradient,
            relation_head_changed=head_changed,
        )
    if paired.fingerprint(model.bundle(optimizer)) != before:
        raise ValueError("same-arm clone changed parent bundle or RNG")
    return result


def save(model, optimizer, path, seed, arm, completed=0, *, protocol=PROTOCOL):
    payload = dict(
        schema="n1-group-relative-paired-checkpoint-v1",
        protocol=protocol,
        update_policy=deepcopy(update.POLICY),
        seed=seed,
        arm=arm,
        completed_updates=completed,
        candidate=model.bundle(optimizer),
        expected_signature=signature(model, optimizer, seed, arm, completed),
    )
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def load(payload, *, protocol=PROTOCOL, allowed=(0,)):
    if (
        payload.get("schema") != "n1-group-relative-paired-checkpoint-v1"
        or payload.get("protocol") != protocol
        or payload.get("update_policy") != update.POLICY
        or payload.get("seed") not in SEEDS
        or payload.get("arm") not in ARMS
        or payload.get("completed_updates") not in allowed
    ):
        raise ValueError("frozen paired update marker, phase and identity required")
    model = runner.transfer.LocalRelationEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(deepcopy(payload["candidate"]["optimizer"]))
    completed = payload["completed_updates"]
    if len(optimizer.state) != (35 if completed == 0 else 41):
        raise ValueError("full old/new Adam state required")
    for name, parameter in model.named_parameters():
        if parameter in optimizer.state and int(
            optimizer.state[parameter]["step"]
        ) != runner.expected_step(name, payload["arm"], completed):
            raise ValueError("per-parameter restored arm steps differ")
    update.bounded.grouped_parameters(model)
    return model, optimizer


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = load(payload)
    return dict(
        signature=signature(model, optimizer, payload["seed"], payload["arm"]),
        cpu_seconds=time.process_time(),
        signature_clone_updates=1,
    )


def independent(path, expected, budget):
    budget.guard(reserve=10)
    try:
        output = subprocess.run(
            [sys.executable, str(Path(__file__)), "--restore-worker", str(path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=min(30, budget.limit - budget.used() - 2),
            env=dict(
                os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
            ),
        )
        result = json.loads(output.stdout)
    except Exception:
        budget.accounting_complete = False
        raise
    budget.children += result["cpu_seconds"]
    budget.guard()
    if result["signature"] != expected or result["signature_clone_updates"] != 1:
        raise ValueError("independent full same-arm bounded recovery differs")
    return dict(path=str(path), sha256=paired.sha(path), equal=True, **result)


def readiness():
    if paired.sha(calibration.REPORT) != CALIBRATION_SHA:
        raise ValueError("frozen68 calibration report required")
    record = json.loads(calibration.REPORT.read_text(encoding="utf-8"))
    if (
        not record["complete"]
        or not record["training_sufficient"]
        or record["code_identity"] != calibration.identity()
        or not runner.sufficient(record["milestones"][-1])
        or record["confirmation_queries"] != 0
    ):
        raise ValueError("complete unchanged68 with original four gates required")
    _, guards, _ = calibration.readiness()
    guards[str(calibration.REPORT)] = CALIBRATION_SHA
    for check in record["restore_checks"]:
        if not check["equal"] or paired.sha(Path(check["path"])) != check["sha256"]:
            raise ValueError("68 checkpoint byte guards or recovery changed")
        guards[check["path"]] = check["sha256"]
    ready, seals = runner.readiness()
    return ready, guards, seals


def audit():
    folder = (
        Path(get_external_path("data/checkpoints")) / "n1_group_relative_paired_preflight_20261015"
    )
    if REPORT.exists() or folder.exists():
        raise FileExistsError("exclusive nine-model engineering; no restart")
    budget = paired.CpuBudget(180)
    ready, guards, seals = readiness()
    _, source, _ = runner.transfer.source_payload()
    folder.mkdir(exist_ok=False)
    result = dict(
        protocol=PROTOCOL,
        code_identity=identity(),
        source_guards=guards,
        confirmation_seals=seals,
        initial_fingerprints={str(seed): [] for seed in SEEDS},
        restore_checks=[],
        signature_clone_updates=0,
        main_trajectory_updates=0,
        curriculum_queries=0,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
        can_promote=False,
        status="running",
    )
    try:
        for seed in SEEDS:
            for arm in ARMS:
                budget.guard(reserve=10)
                model, optimizer = runner.fresh(source, seed, ready)
                result["initial_fingerprints"][str(seed)].append(
                    paired.fingerprint(model.bundle(optimizer))
                )
                path = folder / f"s{seed}_{arm}_zero.pt"
                payload = save(model, optimizer, path, seed, arm)
                result["signature_clone_updates"] += 1
                result["restore_checks"].append(
                    independent(path, payload["expected_signature"], budget)
                )
                result["signature_clone_updates"] += 1
                print(
                    f"bounded69 seed={seed} arm={arm} restored cpu={budget.used():.2f}", flush=True
                )
        if any(
            len(values) != 3 or len(set(values)) != 1
            for values in result["initial_fingerprints"].values()
        ):
            raise ValueError("same-seed complete initial bundles must pair exactly")
        result["status"] = "complete"
    except Exception as error:
        result.update(status="stopped", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no restart")
    finally:
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            sources_unchanged=all(paired.sha(Path(p)) == digest for p, digest in guards.items()),
            code_unchanged=identity() == result["code_identity"],
        )
        result["complete"] = (
            result["status"] == "complete"
            and len(result["restore_checks"]) == 9
            and result["signature_clone_updates"] == 18
            and result["cpu_seconds"] <= 180
            and all(
                result[k]
                for k in ("sources_unchanged", "code_unchanged", "cpu_accounting_complete")
            )
        )
        result["synthetic_qa_sequences"] = 2 * result["signature_clone_updates"]
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
        result = audit()
        print(json.dumps({k: result.get(k) for k in ("complete", "cpu_seconds", "error")}))
        raise SystemExit(0 if result["complete"] else 1)
    else:
        print(json.dumps(dict(read_only=True, protocol=PROTOCOL)))
