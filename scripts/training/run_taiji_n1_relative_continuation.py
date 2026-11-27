"""Approved PLAN32/SPEC33: fixed equal continuation, with first v3 holdout still gated."""

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

from scripts.training import run_taiji_n1_native_evidence_paired as paired  # noqa: E402
from scripts.training import run_taiji_n1_relative_paired as source  # noqa: E402
from scripts.training import run_taiji_n1_relative_read as pilot  # noqa: E402
from scripts.training import run_taiji_n1_v3_train_calibration as calibration  # noqa: E402
from scripts.training.audit_taiji_n1_native_evidence_train_fit import summarize  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.native_evidence_fidelity import learning_loss  # noqa: E402
from taiji.native_relative_evidence_candidate import RelativeEvidenceCandidate  # noqa: E402

REPORT = ROOT / "reports/taiji_n1_relative_continuation_20261015.json"
PREFLIGHT = ROOT / "reports/taiji_n1_relative_continuation_preflight_20261015.json"
BASELINE = source.REPORT
BASELINE_SHA = "b10fe3092c9fa6abd08f64fed48f02e59c4261a20c58a49a85f9702d0c5e06dd"
RUN_ID = "n1_relative_continuation_20261015"
PROTOCOL = {
    **source.PROTOCOL,
    "schema": "n1-relative-continuation-v1",
    "initialization": "restore-all-nine-final-adam-rng",
    "starting_updates_per_model": 6144,
    "updates_per_model": 3072,
    "cumulative_updates_per_model": 9216,
    "total_updates": 27648,
    "cpu_limit_seconds": 600,
    "cpu_includes_engineering_preflight": True,
    "baseline_sha256": BASELINE_SHA,
}


def identity():
    return {
        "protocol": PROTOCOL,
        "pilot_code": pilot.code_identity(),
        "source_code": source.identity(),
        "runner_sha256": paired.sha(Path(__file__)),
    }


def checked_source():
    if paired.sha(BASELINE) != BASELINE_SHA:
        raise ValueError("frozen source report changed")
    record = json.loads(BASELINE.read_text(encoding="utf-8"))
    if (
        record["identity"] != source.identity()
        or not record["complete"]
        or record["status"] != "training-insufficient"
        or record["completed_updates"] != 55296
        or record["holdout_queries"] != 0
        or record["dev_queries"] != 0
        or len(record["restore_checks"]) != 18
        or not all(c["equal"] for c in record["restore_checks"])
    ):
        raise ValueError("all nine complete frozen source models and no prior heldout required")
    expected = {(s, a) for s in paired.SEEDS for a in paired.ARMS}
    if (
        len(record["frozen_models"]) != 9
        or {(e["seed"], e["arm"]) for e in record["frozen_models"]} != expected
    ):
        raise ValueError("nine unique source model identities required")
    for seed in paired.SEEDS:
        initial = {t["initial_parameters"] for t in record["training"] if t["seed"] == seed}
        if len(initial) != 1:
            raise ValueError("source arms lack paired initial history")
    return record


def restore_input(payload, seed, arm):
    if (
        payload["protocol"] != source.PROTOCOL
        or payload["completed_updates"] != 6144
        or payload["seed"] != seed
        or payload["arm"] != arm
    ):
        raise ValueError("source input checkpoint protocol, seed, arm or update mismatch")
    model = RelativeEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    if pilot.signature(model, optimizer) != payload["expected_signature"]:
        raise ValueError("source input signature or complete optimizer continuation differs")
    return model, optimizer


def fresh(seed, train, arm="position"):
    record = checked_source()
    entry = next(e for e in record["frozen_models"] if e["seed"] == seed and e["arm"] == arm)
    path = Path(entry["path"])
    if paired.sha(path) != entry["sha256"]:
        raise ValueError("source final checkpoint hash changed")
    model, optimizer = restore_input(
        torch.load(path, map_location="cpu", weights_only=True), seed, arm
    )
    if paired.fingerprint(model.state_dict()) != entry["parameters"]:
        raise ValueError("source parameters differ from frozen final")
    return model, optimizer


def save(model, optimizer, path, seed, arm, completed):
    payload = {
        "protocol": PROTOCOL,
        "seed": seed,
        "arm": arm,
        "completed_updates": completed,
        "candidate": model.bundle(optimizer),
        "expected_signature": pilot.signature(model, optimizer),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def restore(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["protocol"] != PROTOCOL
        or payload["seed"] not in paired.SEEDS
        or payload["arm"] not in paired.ARMS
    ):
        raise ValueError("paired relative checkpoint protocol mismatch")
    model = RelativeEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    return {"signature": pilot.signature(model, optimizer), "cpu_seconds": time.process_time()}


def independent(path, expected, budget):
    budget.guard(reserve=10)
    try:
        output = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
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
    if result["signature"] != expected:
        raise ValueError("independent relative paired restoration differs")
    return {"path": str(path), "sha256": paired.sha(path), **result, "equal": True}


def train_step(model, optimizer, item, session, arm):
    if item["split"] != "train" or arm not in paired.ARMS:
        raise ValueError("paired updates require train and a frozen arm")
    paired.observe(model, paired.project(item), session)
    optimizer.zero_grad(set_to_none=True)
    loss = learning_loss(
        model, session, item["question"].encode(), item["answer"].encode(), arm=arm
    )
    if not torch.isfinite(loss.total):
        raise ValueError("nonfinite paired loss")
    loss.total.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite paired gradient")
    optimizer.step()
    return {name: float(getattr(loss, name).detach()) for name in ("total", "qa", "observation")}


@torch.no_grad()
def measure(model, item, arm, session, *, holdout_ready=False):
    if item["split"] not in ("train", "holdout") or (
        item["split"] == "holdout" and not holdout_ready
    ):
        raise ValueError(
            "heldout query requires all frozen models and sufficient position training; dev forbidden"
        )
    paired.observe(model, paired.project(item), session)
    tape = paired.evidence(model, session, arm)
    sources = model.sources(session)
    target = item["answer"].encode()
    labels = torch.tensor([*target, 256])
    logits, _ = model.teacher_logits(tape, session, item["question"].encode(), target)
    trace = model.generate_trace(tape, session, item["question"].encode(), 64)
    if model.sources(session) != sources:
        raise ValueError("evaluation changed external observations")
    return {
        "id": item["id"],
        "split": item["split"],
        "kind": item["kind"],
        "target_symbols": labels.tolist(),
        "teacher_symbols": logits.argmax(dim=-1).tolist(),
        "teacher_tokens_correct": int((logits.argmax(dim=-1) == labels).sum()),
        "teacher_exact": torch.equal(logits.argmax(dim=-1), labels),
        "teacher_ce": float(torch.nn.functional.cross_entropy(logits, labels)),
        "raw_hex": trace.raw.hex(),
        "output_hex": trace.output.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "utf8_remaining": trace.utf8_remaining,
        "free_exact": trace.raw == target and trace.stop_reason == "boundary",
    }


def training_ready(training, rows, restores):
    expected = {(seed, arm) for seed in paired.SEEDS for arm in paired.ARMS}
    if len(training) != 9 or {(t["seed"], t["arm"]) for t in training} != expected:
        return False
    keys = [(row["seed"], row["arm"], row["id"]) for row in rows]
    required = {(seed, arm, f"train-{i:03d}") for seed, arm in expected for i in range(192)}
    if len(keys) != len(required) or set(keys) != required:
        return False
    restore_keys = {(c["seed"], c["arm"], c["phase"]) for c in restores}
    if len(restores) != 18 or restore_keys != {
        (s, a, p) for s, a in expected for p in ("zero", "final")
    }:
        return False
    return (
        all(t["updates"] == 9216 for t in training)
        and all(c["equal"] for c in restores)
        and all(
            calibration.sufficient(t["train_summary"]) for t in training if t["arm"] == "position"
        )
    )


def verdict(rows):
    expected = {
        (seed, arm, f"holdout-{i:03d}")
        for seed in paired.SEEDS
        for arm in paired.ARMS
        for i in range(32)
    }
    keys = [(row["seed"], row["arm"], row["id"]) for row in rows]
    scores, kinds = {}, {}
    for seed in paired.SEEDS:
        scores[str(seed)], kinds[str(seed)] = {}, {}
        for arm in paired.ARMS:
            selected = [r for r in rows if r["seed"] == seed and r["arm"] == arm]
            scores[str(seed)][arm] = sum(r["free_exact"] for r in selected)
            kinds[str(seed)][arm] = {
                k: sum(r["free_exact"] for r in selected if r["kind"] == k) for k in paired.KINDS
            }
    complete = len(keys) == len(expected) and set(keys) == expected
    seed_gate = all(scores[str(s)]["position"] >= 16 for s in paired.SEEDS)
    kind_gate = all(kinds[str(s)]["position"][k] >= 2 for s in paired.SEEDS for k in paired.KINDS)
    gaps = {
        arm: sum(scores[str(s)]["position"] - scores[str(s)][arm] for s in paired.SEEDS)
        for arm in ("terminal", "cut")
    }
    return {
        "complete": complete,
        "holdout_exact": scores,
        "holdout_kind_exact": kinds,
        "total_advantage": gaps,
        "mean_advantage": {k: v / 3 for k, v in gaps.items()},
        "each_seed_gate": seed_gate,
        "each_kind_gate": kind_gate,
        "can_advance": complete and seed_gate and kind_gate and all(g >= 12 for g in gaps.values()),
    }


def preflight():
    if PREFLIGHT.exists():
        raise FileExistsError("paired preflight exists; no repeat")
    budget = paired.CpuBudget(limit=60)
    train = pilot.checked_train()
    result = {
        "identity": identity(),
        "rows": [],
        "optimizer_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "status": "running",
        "input_checks": [],
    }
    try:
        for seed in paired.SEEDS:
            for arm in paired.ARMS:
                budget.guard()
                model, _ = fresh(seed, train, arm)
                result["input_checks"].append({"seed": seed, "arm": arm, "equal": True})
        for seed in paired.SEEDS:
            model, _ = fresh(seed, train)
            before = paired.fingerprint(model.state_dict())
            losses = {}
            for arm in paired.ARMS:
                budget.guard()
                model.zero_grad(set_to_none=True)
                loss = learning_loss(
                    model,
                    "zero-probe",
                    train[0]["question"].encode(),
                    train[0]["answer"].encode(),
                    arm=arm,
                )
                loss.total.backward()
                early = loss.tape.states[0].grad
                if arm == "cut":
                    if early is not None or any(
                        p.grad is not None for p in model.writer.parameters()
                    ):
                        raise ValueError("cut writer credit leaked")
                elif early is None or early.norm() <= 0:
                    raise ValueError("early writer credit absent")
                losses[arm] = loss
                result["rows"].append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "loss": float(loss.total.detach()),
                        "early_grad_norm": None if early is None else float(early.norm()),
                    }
                )
            if not torch.equal(losses["position"].total, losses["cut"].total) or not torch.equal(
                losses["position"].observation, losses["terminal"].observation
            ):
                raise ValueError("paired forward/auxiliary mismatch")
            if paired.fingerprint(model.state_dict()) != before:
                raise ValueError("engineering preflight changed parameters")
        model, optimizer = fresh(11, train)
        folder = Path(get_external_path("data/checkpoints")) / f"{RUN_ID}_preflight"
        folder.mkdir(exist_ok=False)
        path = folder / "zero.pt"
        payload = save(model, optimizer, path, 11, "position", 6144)
        result["restore"] = independent(path, payload["expected_signature"], budget)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        result.update(
            cpu_seconds=budget.used(),
            cpu_accounting_complete=budget.accounting_complete,
            code_unchanged=identity() == result["identity"],
        )
        result["passed"] = (
            result["status"] == "complete"
            and len(result["rows"]) == 9
            and len(result["input_checks"]) == 9
            and result["cpu_seconds"] <= 60
            and result["cpu_accounting_complete"]
            and result["code_unchanged"]
        )
        paired.write_new(PREFLIGHT, result)
    return result


def run():
    if REPORT.exists():
        raise FileExistsError("paired formal report exists; no repeat or overwrite")
    frozen = identity()
    ready = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not ready["passed"] or ready["identity"] != frozen or paired.sha(BASELINE) != BASELINE_SHA:
        raise ValueError("matching engineering preflight and frozen calibration required")
    manifest = json.loads(calibration.MANIFEST.read_text(encoding="utf-8"))
    train = pilot.checked_train()
    holdout = [item for item in manifest["episodes"] if item["split"] == "holdout"]
    if len(holdout) != 32 or any(sum(i["kind"] == k for i in holdout) != 8 for k in paired.KINDS):
        raise ValueError("full four-kind holdout required")
    folder = Path(get_external_path("data/checkpoints")) / RUN_ID
    folder.mkdir(exist_ok=False)
    budget = paired.CpuBudget(limit=600)
    budget.children += ready["cpu_seconds"]
    inputs = checked_source()["frozen_models"]
    result = {
        "identity": frozen,
        "preflight_sha256": paired.sha(PREFLIGHT),
        "preflight_cpu_seconds": ready["cpu_seconds"],
        "input_models": inputs,
        "prior_updates": 55296,
        "training": [],
        "train_rows": [],
        "holdout_rows": [],
        "restore_checks": [],
        "completed_updates": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "frozen_models": [],
        "status": "running",
        "can_promote": False,
    }
    models = []
    current = None
    try:
        for seed in paired.SEEDS:
            initial = None
            for arm in paired.ARMS:
                budget.guard(reserve=10)
                model, optimizer = fresh(seed, train, arm)
                current = (model, optimizer, seed, arm, 6144)
                digest = paired.fingerprint(model.state_dict())
                initial = digest
                zero = folder / f"s{seed}_{arm}_zero.pt"
                payload = save(model, optimizer, zero, seed, arm, 6144)
                result["restore_checks"].append(
                    dict(
                        independent(zero, payload["expected_signature"], budget),
                        seed=seed,
                        arm=arm,
                        phase="zero",
                    )
                )
                losses = []
                for step in range(3072):
                    budget.guard()
                    losses.append(
                        train_step(
                            model,
                            optimizer,
                            train[(6144 + step) % len(train)],
                            f"continuation-train-{6144 + step}",
                            arm,
                        )
                    )
                    current = (model, optimizer, seed, arm, 6144 + step + 1)
                    result["completed_updates"] += 1
                before = paired.fingerprint(model.state_dict())
                rows = []
                for item in train:
                    budget.guard()
                    rows.append(
                        dict(
                            measure(model, item, arm, f"continuation-eval-{item['id']}"),
                            seed=seed,
                            arm=arm,
                        )
                    )
                if paired.fingerprint(model.state_dict()) != before:
                    raise ValueError("train evaluation changed parameters")
                result["train_rows"].extend(rows)
                summary = summarize(rows)[str(seed)][arm]
                final = folder / f"s{seed}_{arm}_final.pt"
                payload = save(model, optimizer, final, seed, arm, 9216)
                result["restore_checks"].append(
                    dict(
                        independent(final, payload["expected_signature"], budget),
                        seed=seed,
                        arm=arm,
                        phase="final",
                    )
                )
                result["training"].append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "updates": 9216,
                        "additional_updates": 3072,
                        "initial_parameters": initial,
                        "final_parameters": before,
                        "train_summary": summary,
                        "losses": losses,
                    }
                )
                result["frozen_models"].append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "path": str(final),
                        "sha256": paired.sha(final),
                        "parameters": before,
                    }
                )
                models.append((seed, arm, model))
                print(
                    f"trained seed={seed} arm={arm} exact={summary['free_exact']}/192 cpu={budget.used():.2f}",
                    flush=True,
                )
        if identity() != frozen:
            raise ValueError("code drift before first holdout")
        result["training_sufficient"] = training_ready(
            result["training"], result["train_rows"], result["restore_checks"]
        )
        if not result["training_sufficient"]:
            result["status"] = "training-insufficient"
        else:
            for entry in result["frozen_models"]:
                if paired.sha(Path(entry["path"])) != entry["sha256"]:
                    raise ValueError("final checkpoint changed before first holdout")
            result["pre_holdout_freeze"] = {
                "identity": identity(),
                "models": result["frozen_models"],
                "cpu_seconds": budget.used(),
                "completed_updates": result["completed_updates"],
                "restore_count": len(result["restore_checks"]),
            }
            paired.write_new(folder / "pre_holdout_freeze.json", result["pre_holdout_freeze"])
            for seed, arm, model in models:
                before = paired.fingerprint(model.state_dict())
                for item in holdout:
                    budget.guard()
                    row = measure(model, item, arm, f"holdout-{item['id']}", holdout_ready=True)
                    result["holdout_queries"] += 1
                    result["holdout_rows"].append(dict(row, seed=seed, arm=arm))
                if before != paired.fingerprint(model.state_dict()):
                    raise ValueError("holdout changed parameters")
            result["status"] = "complete"
    except paired.BudgetStop as error:
        result.update(status="budget-stop", error=str(error))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interruption; no automatic restart")
    finally:
        if result["status"] not in ("complete", "training-insufficient") and current is not None:
            model, optimizer, seed, arm, completed = current
            try:
                path = folder / f"s{seed}_{arm}_partial.pt"
                save(model, optimizer, path, seed, arm, completed)
                result["partial_checkpoint"] = {"path": str(path), "sha256": paired.sha(path)}
            except Exception as error:
                result["partial_checkpoint_error"] = str(error)
        result.update(
            cpu_seconds=budget.used(),
            cumulative_updates=55296 + result["completed_updates"],
            source_checkpoints_unchanged=all(
                paired.sha(Path(e["path"])) == e["sha256"] for e in inputs
            ),
            cpu_accounting_complete=budget.accounting_complete,
            code_unchanged=identity() == frozen,
            manifest_unchanged=paired.sha(calibration.MANIFEST) == calibration.MANIFEST_SHA,
            baseline_unchanged=paired.sha(BASELINE) == BASELINE_SHA,
            checkpoints_unchanged=all(
                paired.sha(Path(e["path"])) == e["sha256"] for e in result["frozen_models"]
            ),
        )
        result["verdict"] = verdict(result["holdout_rows"])
        result["complete"] = (
            result["status"] in ("complete", "training-insufficient")
            and result["completed_updates"] == 27648
            and len(result["train_rows"]) == 1728
            and len(result["restore_checks"]) == 18
            and all(c["equal"] for c in result["restore_checks"])
            and (
                (
                    result["status"] == "complete"
                    and result["verdict"]["complete"]
                    and result["holdout_queries"] == 288
                )
                or (result["status"] == "training-insufficient" and result["holdout_queries"] == 0)
            )
            and result["cpu_seconds"] <= 600
            and all(
                result[k]
                for k in (
                    "cpu_accounting_complete",
                    "code_unchanged",
                    "manifest_unchanged",
                    "baseline_unchanged",
                    "checkpoints_unchanged",
                    "source_checkpoints_unchanged",
                )
            )
        )
        result["verdict"]["can_advance"] &= result["complete"] and result["status"] == "complete"
        paired.write_new(REPORT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-worker", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        print(json.dumps(restore(args.restore_worker)))
    elif args.preflight:
        output = preflight()
        print(json.dumps(output))
        raise SystemExit(0 if output["passed"] else 1)
    elif args.execute_approved:
        output = run()
        print(json.dumps({k: output[k] for k in ("complete", "status", "cpu_seconds", "verdict")}))
        raise SystemExit(0 if output["complete"] else 1)
    else:
        print(json.dumps({"read_only": True, "protocol": PROTOCOL}))
