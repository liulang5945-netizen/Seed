"""PLAN-N1-14 approved paired pilot; defaults to a read-only recipe check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402
import torch  # noqa: E402

from scripts.training.build_taiji_n1_evidence_curriculum import KINDS, validate  # noqa: E402
from seed_platform.paths import get_external_path  # noqa: E402
from taiji.contracts import Observation  # noqa: E402
from taiji.native_evidence_candidate import NativeEvidenceCandidate  # noqa: E402

MANIFEST = ROOT / "plans/manifests/n1_native_evidence_curriculum_v2_20261015.json"
MANIFEST_SHA = "e95a2e3be6a3a4abca92ea583cf35522545961280b575e5d5373857973d5e979"
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
UPDATES = 384
CPU_LIMIT = 300.0
RUN_ID = "n1_native_evidence_paired_20261015"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    def normalized(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            return {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            }
        if isinstance(item, dict):
            return {str(key): normalized(v) for key, v in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalized(v) for v in item]
        if isinstance(item, bytes):
            return {"bytes_hex": item.hex()}
        return item

    return hashlib.sha256(
        json.dumps(normalized(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def recipe():
    return {
        "schema": "n1-native-evidence-paired-v1",
        "manifest_sha256": MANIFEST_SHA,
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "updates_per_model": UPDATES,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "max_length": 64,
        "dimensions": {"embedding": 16, "hidden": 32, "key": 16},
        "cpu_limit_seconds": CPU_LIMIT,
        "gates": {
            "position_each_seed_exact": 16,
            "position_each_kind": 2,
            "mean_advantage_each_control": 4,
        },
    }


def identity():
    paths = (
        Path(__file__),
        ROOT / "taiji/native_evidence_candidate.py",
        ROOT / "taiji/position_evidence_reference.py",
        ROOT / "taiji/contracts.py",
        ROOT / "taiji/utf8_state.py",
        ROOT / "scripts/training/build_taiji_n1_evidence_curriculum.py",
    )
    return {
        "recipe": recipe(),
        "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in paths},
        "python": sys.version,
        "torch": torch.__version__,
    }


def checked_manifest(path=MANIFEST):
    if sha(path) != MANIFEST_SHA:
        raise ValueError("frozen v2 manifest digest mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate(manifest)
    return manifest


@dataclass(frozen=True)
class EpisodeInput:
    observations: tuple[str, ...]
    question: bytes


def project(item):
    """No label, kind, item ID or split enters this inference payload."""
    return EpisodeInput(tuple(item["observations"]), item["question"].encode("utf-8"))


def observe(model, data, session):
    for index, text in enumerate(data.observations, 1):
        model.observe(session, Observation("text", text, index, f"source-{index}"))


def evidence(model, session, arm):
    if arm not in ARMS:
        raise ValueError("unknown paired arm")
    return model.evidence(session, cut_credit=arm == "cut", terminal_per_source=arm == "terminal")


def train_step(model, optimizer, item, session, arm):
    if item["split"] != "train":
        raise ValueError("only train episodes may update parameters")
    data = project(item)
    observe(model, data, session)
    optimizer.zero_grad(set_to_none=True)
    logits, _ = model.teacher_logits(
        evidence(model, session, arm), session, data.question, item["answer"].encode()
    )
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([*item["answer"].encode(), 256]))
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    loss.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("nonfinite gradient")
    optimizer.step()
    return float(loss.detach())


class BudgetStop(RuntimeError):
    pass


class CpuBudget:
    def __init__(self, limit=CPU_LIMIT, clock=time.process_time):
        self.limit, self.clock = limit, clock
        self.children = 0.0
        self.accounting_complete = True

    def used(self, active_child=0.0):
        return self.clock() + self.children + active_child

    def guard(self, active_child=0.0, reserve=2.0):
        if self.used(active_child) + reserve >= self.limit:
            raise BudgetStop("cumulative parent/worker CPU limit reached")


def checkpoint(model, optimizer, path, seed, arm, completed, session, probe):
    payload = {
        "candidate": model.bundle(optimizer),
        "seed": seed,
        "arm": arm,
        "completed_updates": completed,
        "recipe_sha256": fingerprint(recipe()),
        "session": session,
        "probe": project(probe).question,
    }
    with torch.no_grad():
        logits, _, _ = model.step(
            evidence(model, session, arm), session, model.question_state(payload["probe"]), 256
        )
    payload["expected_signature"] = {
        "candidate": fingerprint(payload["candidate"]),
        "first_logits": fingerprint(logits),
    }
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return payload


def restore_signature(payload):
    if payload["recipe_sha256"] != fingerprint(recipe()):
        raise ValueError("checkpoint recipe mismatch")
    model = NativeEvidenceCandidate.restore_bundle(payload["candidate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(payload["candidate"]["optimizer"])
    restored = fingerprint(model.bundle(optimizer))
    # Probe uses a train question and own first-step distribution, never holdout labels.
    with torch.no_grad():
        logits, _, _ = model.step(
            evidence(model, payload["session"], payload["arm"]),
            payload["session"],
            model.question_state(payload["probe"]),
            256,
        )
    return {"candidate": restored, "first_logits": fingerprint(logits)}


def independent_restore(path, budget):
    budget.guard()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, str(Path(__file__).resolve()), "--restore-worker", str(path)]
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
    )
    observed = 0.0
    try:
        monitored = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                cpu = monitored.cpu_times()
                observed = max(observed, cpu.user + cpu.system)
            except psutil.NoSuchProcess:
                pass
            budget.guard(observed)
            time.sleep(0.025)
        stdout, stderr = process.communicate()
        if process.returncode:
            raise RuntimeError(f"restore worker failed: {stderr[-2000:]}")
        result = json.loads(stdout)
        budget.children += max(observed, result["cpu_seconds"])
        observed = 0.0
        budget.guard()
        return result
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.communicate()
        if observed:
            budget.children += observed
        budget.accounting_complete = False
        raise


def checked_restore(payload, path, budget):
    expected = payload["expected_signature"]
    actual = independent_restore(path, budget)
    if actual["signature"] != expected:
        raise ValueError(
            "independent weights-only model/optimizer/source/RNG/logits restore mismatch"
        )
    return {
        "checkpoint": str(path),
        "sha256": sha(path),
        "signature": expected,
        "cpu_seconds": actual["cpu_seconds"],
        "equal": True,
    }


def evaluate(model, item, session, arm):
    if item["split"] not in ("dev", "holdout"):
        raise ValueError("evaluation only accepts frozen dev or holdout")
    before = fingerprint(model.state_dict())
    data = project(item)
    observe(model, data, session)
    sources_before = model.sources(session)
    with torch.no_grad():
        trace = model.generate_trace(evidence(model, session, arm), session, data.question, 64)
    if fingerprint(model.state_dict()) != before or model.sources(session) != sources_before:
        raise ValueError("query generation mutated parameters or wrote evidence")
    return {
        "id": item["id"],
        "split": item["split"],
        "kind": item["kind"],
        "raw_hex": trace.raw.hex(),
        "output_hex": trace.output.hex(),
        "stop_reason": trace.stop_reason,
        "generated_steps": trace.generated_steps,
        "utf8_remaining": trace.utf8_remaining,
        "expected_hex": item["answer"].encode().hex(),
        "exact": trace.raw == item["answer"].encode() and trace.stop_reason == "boundary",
    }


def verdict(rows, completed_updates):
    expected = {
        (seed, arm, f"{split}-{i:03d}")
        for seed in SEEDS
        for arm in ARMS
        for split in ("dev", "holdout")
        for i in range(32)
    }
    keys = [(r["seed"], r["arm"], r["id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate paired evaluation")
    complete = set(keys) == expected and completed_updates == len(SEEDS) * len(ARMS) * UPDATES
    scores, kinds = {}, {}
    for seed in SEEDS:
        scores[str(seed)], kinds[str(seed)] = {}, {}
        for arm in ARMS:
            selected = [
                r for r in rows if r["seed"] == seed and r["arm"] == arm and r["split"] == "holdout"
            ]
            scores[str(seed)][arm] = sum(r["exact"] for r in selected)
            kinds[str(seed)][arm] = {
                kind: sum(r["exact"] for r in selected if r["kind"] == kind) for kind in KINDS
            }
    each_seed = all(scores[str(s)]["position"] >= 16 for s in SEEDS)
    each_kind = all(kinds[str(s)]["position"][k] >= 2 for s in SEEDS for k in KINDS)
    gaps = {
        arm: sum(scores[str(s)]["position"] - scores[str(s)][arm] for s in SEEDS) / len(SEEDS)
        for arm in ("terminal", "cut")
    }
    return {
        "complete": complete,
        "holdout_exact": scores,
        "holdout_kind_exact": kinds,
        "mean_advantage": gaps,
        "each_seed_gate": each_seed,
        "each_kind_gate": each_kind,
        "can_advance": complete and each_seed and each_kind and all(v >= 4 for v in gaps.values()),
    }


def run(*, approved=False):
    if not approved:
        raise PermissionError(
            "explicitly approved PLAN14 execution required; default is check-only"
        )
    manifest = checked_manifest()
    frozen = identity()
    report_path = ROOT / "reports" / f"taiji_{RUN_ID}_final.json"
    if report_path.exists():
        raise FileExistsError("formal report exists; no automatic rerun")
    base = Path(get_external_path("data/checkpoints"))
    folder = base / RUN_ID
    folder.mkdir(exist_ok=False)
    write_new(folder / "identity.json", frozen)
    budget = CpuBudget()
    result = {
        "identity": frozen,
        "status": "running",
        "rows": [],
        "restore_checks": [],
        "training": [],
        "completed_updates": 0,
        "capability_claim": False,
    }
    models = []
    current = None
    try:
        train = [item for item in manifest["episodes"] if item["split"] == "train"]
        for seed in SEEDS:
            initial = None
            for arm in ARMS:
                budget.guard()
                torch.manual_seed(seed)
                model = NativeEvidenceCandidate()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                current = (model, optimizer, seed, arm, 0)
                digest = fingerprint(model.state_dict())
                if initial is not None and initial != digest:
                    raise ValueError("paired arms have different initial weights")
                initial = digest
                observe(model, project(train[0]), "zero-probe")
                zero_path = folder / f"s{seed}_{arm}_zero.pt"
                payload = checkpoint(
                    model, optimizer, zero_path, seed, arm, 0, "zero-probe", train[0]
                )
                result["restore_checks"].append(checked_restore(payload, zero_path, budget))
                losses = []
                for step in range(UPDATES):
                    budget.guard()
                    losses.append(
                        train_step(model, optimizer, train[step % len(train)], f"train-{step}", arm)
                    )
                    current = (model, optimizer, seed, arm, step + 1)
                    result["completed_updates"] += 1
                trained_path = folder / f"s{seed}_{arm}_trained.pt"
                budget.guard()
                payload = checkpoint(
                    model, optimizer, trained_path, seed, arm, UPDATES, "train-383", train[-1]
                )
                result["restore_checks"].append(checked_restore(payload, trained_path, budget))
                result["training"].append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "updates": UPDATES,
                        "initial_parameters": initial,
                        "final_parameters": fingerprint(model.state_dict()),
                        "losses": losses,
                    }
                )
                models.append((seed, arm, model))
                print(f"trained seed={seed} arm={arm} cpu={budget.used():.2f}", flush=True)
        # No dev or holdout model query occurs until all nine final models freeze.
        if identity() != frozen:
            raise ValueError("code or recipe drift before evaluation")
        for seed, arm, model in models:
            for item in manifest["episodes"]:
                if item["split"] == "train":
                    continue
                budget.guard()
                row = evaluate(model, item, f"eval-{item['id']}", arm)
                result["rows"].append(dict(row, seed=seed, arm=arm))
        if identity() != frozen:
            raise ValueError("code or recipe drift during evaluation")
        result["status"] = "complete"
    except BudgetStop as error:
        result.update(status="budget-stop", error=str(error))
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
    except KeyboardInterrupt:
        result.update(status="interrupted", error="operator interrupted; no automatic restart")
    finally:
        if current is not None and result["status"] != "complete":
            model, optimizer, seed, arm, completed = current
            path = folder / f"s{seed}_{arm}_partial.pt"
            try:
                checkpoint(model, optimizer, path, seed, arm, completed, "zero-probe", train[0])
                result["partial_checkpoint"] = {"path": str(path), "sha256": sha(path)}
            except Exception as error:
                result["partial_checkpoint_error"] = str(error)
        result["cpu_seconds"] = budget.used()
        result["cpu_accounting_complete"] = budget.accounting_complete
        result["cpu_within_limit"] = budget.used() <= CPU_LIMIT
        result["verdict"] = verdict(result["rows"], result["completed_updates"])
        result["restore_checks_complete"] = len(result["restore_checks"]) == 18 and all(
            c["equal"] for c in result["restore_checks"]
        )
        result["verdict"]["can_advance"] &= (
            result["status"] == "complete"
            and result["cpu_within_limit"]
            and budget.accounting_complete
            and result["restore_checks_complete"]
        )
        result["cpu_measurement"] = (
            "parent process_time incl imports plus each worker max(process_time,last psutil CPU); endpoint before final report serialization"
        )
        write_new(report_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-approved", action="store_true")
    parser.add_argument("--restore-worker", type=Path)
    parser.add_argument("--out-report", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.restore_worker:
        payload = torch.load(args.restore_worker, map_location="cpu", weights_only=True)
        output = {"signature": restore_signature(payload), "cpu_seconds": time.process_time()}
    elif args.execute_approved:
        if args.out_report:
            parser.error("formal output paths are fixed")
        output = run(approved=True)
        print(
            json.dumps(
                {
                    "status": output["status"],
                    "cpu_seconds": output["cpu_seconds"],
                    "verdict": output["verdict"],
                }
            )
        )
        return 0 if output["status"] == "complete" else 1
    else:
        manifest = checked_manifest()
        output = {
            "identity": identity(),
            "counts": validate(manifest),
            "formal_execution": False,
            "optimizer_steps": 0,
            "model_queries": 0,
        }
        if args.out_report:
            write_new(args.out_report, output)
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
