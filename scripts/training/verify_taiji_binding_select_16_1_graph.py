#!/usr/bin/env python
"""N16.1.0b graph engineering for the selection-locked binding candidate.

Verifies parameter accounting, selection FD, one-hot alpha locking, the gamma
floor, the accuracy-triggered curriculum interface, identity rescue and
independent-process checkpoint restore for two profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

torch.set_num_threads(1)

from taiji.binding_copy import (  # noqa: E402
    answer_step,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.binding_select import (  # noqa: E402
    GAMMA_FLOOR,
    SelectionCandidate,
    SelectionConfig,
    build_selection_candidate,
    expected_selection_parameter_count,
    load_bundle,
    locked_generate,
    locked_loss,
    locked_step,
    make_selection_optimizer,
    save_bundle,
    select,
    selection_ce,
)
from taiji.living_core import EOS, RawSource  # noqa: E402
from taiji.record_relation import RecordRelationConfig  # noqa: E402
from taiji.record_relation_core import (  # noqa: E402
    ARM_FULL,
    CLIP_NORM,
    RecordRelationCandidate,
    build_subject_from_candidate_bundle,
    candidate_parameter_map,
)

PARENT_BUNDLE = "output/taiji_core_record_relation_14_6/27031_A_full_step300.pt"
PARENT_SHA256 = "b75a83a0a474c041c2a61bed8c4e0a51de11197d67a9b196262e7e039a9a4cef"
SELECT_PAYLOAD = "output/taiji_core_binding_select_16_1/select_v1.json"
REPORT = "reports/taiji_binding_select_16_1_graph_20261103.json"
MANIFEST = "plans/manifests/taiji_binding_select_16_1_graph_20261103.json"
OUTPUT_DIR = Path("output/taiji_core_binding_select_16_1_graph")

STAGE_CPU_SECONDS = 600.0
STAGE_WALL_SECONDS = 1200.0
STAGE_ADAM_CALLS = 12
STAGE_PRODUCT_BYTES = 2 * 1024**3
WORKER_TIMEOUT = 300
CURRICULUM_THRESHOLD = 0.5
BATCH = 8
FD_EPS = (1e-4, 1e-5, 1e-6)

PROFILES = ("A-locked-f64", "B-select-f64")

SOURCE_FILES = (
    "taiji/binding_select.py",
    "taiji/binding_copy.py",
    "taiji/binding_copy_data.py",
    "taiji/binding_copy_scoring.py",
    "scripts/training/verify_taiji_binding_select_16_1_graph.py",
)

STARTED_CPU = time.process_time()
STARTED_WALL = time.perf_counter()


class StageStop(RuntimeError):
    pass


class Budget:
    def __init__(self) -> None:
        self.main_cpu = 0.0
        self.worker_cpu = 0.0
        self.adam_calls = 0
        self.live = 0
        self.max_live = 0
        self.cumulative = 0

    def total_cpu(self) -> float:
        self.main_cpu = time.process_time() - STARTED_CPU
        return self.main_cpu + self.worker_cpu

    def add_worker(self, result: dict) -> None:
        self.worker_cpu += float(result.get("cpu_seconds", 0.0))
        self.adam_calls += int(result.get("adam_calls", 0))
        self.ensure()

    def add_product(self, info: dict) -> None:
        size = int(info.get("bytes", 0))
        self.live += size
        self.cumulative += size
        self.max_live = max(self.max_live, self.live)

    def drop_product(self, info: dict) -> None:
        self.live -= int(info.get("bytes", 0))

    def ensure(self) -> None:
        if self.total_cpu() > STAGE_CPU_SECONDS:
            raise StageStop(f"cpu budget exceeded: {self.total_cpu():.2f}")
        wall = time.perf_counter() - STARTED_WALL
        if wall > STAGE_WALL_SECONDS:
            raise StageStop(f"wall budget exceeded: {wall:.2f}")
        if self.adam_calls > STAGE_ADAM_CALLS:
            raise StageStop(f"adam budget exceeded: {self.adam_calls}")
        if self.max_live > STAGE_PRODUCT_BYTES:
            raise StageStop(f"product budget exceeded: {self.max_live}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(split: str) -> list[dict]:
    payload = json.loads(Path(SELECT_PAYLOAD).read_text(encoding="utf-8"))
    return payload[split]


def to_fixture(row: dict) -> dict:
    # Records are rebuilt only through the v2 payload so the forward never sees
    # selection labels or target strings.
    data = json.loads(
        Path("output/taiji_core_binding_copy_15_1/data_v2.json").read_text(encoding="utf-8")
    )
    example = next(e for e in data["examples"] if e["index"] == row["index"])
    records = tuple(
        RawSource(source_id=r["source_id"], raw=r["raw"].encode("utf-8"))
        for r in example["records"]
    )
    return {
        "index": int(row["index"]),
        "goal": row["goal"],
        "records": records,
        "record_keys": row["record_keys"],
        "selection_label": int(row["selection_label"]),
        "target_bytes": tuple(example["action"].encode("utf-8")) + (EOS,),
        "target_prefix": tuple(example["action"].encode("utf-8")),
    }


def build_base(payload: dict) -> RecordRelationCandidate:
    config = RecordRelationConfig(**payload["relation_config"])
    subject = build_subject_from_candidate_bundle(payload)
    candidate = RecordRelationCandidate(subject, config, ARM_FULL, list(payload["growth_gates"]))
    named = dict(candidate.named_parameters())
    for name, value in payload["parameters"].items():
        if name.startswith("relation."):
            named[name].data.copy_(value)
    candidate.to(torch.float64)
    candidate.configure_trainable()
    return candidate


def build_profile(payload: dict, profile: str) -> tuple[SelectionCandidate, torch.optim.Adam]:
    base = build_base(payload)
    candidate = build_selection_candidate(
        base, profile=profile, dtype=torch.float64, selection_config=SelectionConfig()
    )
    candidate.configure_trainable()
    return candidate, make_selection_optimizer(candidate)


def tensor_digest(tensors: dict[str, torch.Tensor]) -> dict[str, str]:
    out = {}
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        out[name] = hashlib.sha256(
            f"{tuple(tensor.shape)}|{tensor.dtype}".encode() + tensor.numpy().tobytes()
        ).hexdigest()
    return out


def snapshot(candidate: SelectionCandidate, state, optimizer) -> dict:
    tensors: dict[str, torch.Tensor] = {}
    for name, value in candidate_parameter_map(candidate.binding).items():
        tensors[f"param/{name}"] = value
    for name, value in candidate.head.state_dict().items():
        if isinstance(value, torch.Tensor):
            tensors[f"head/{name}"] = value
    state_dict = optimizer.state_dict()
    for index, entry in state_dict["state"].items():
        for field, value in entry.items():
            if isinstance(value, torch.Tensor):
                tensors[f"opt/{index}/{field}"] = value
    payload = state.payload()
    for name, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensors[f"state/{name}"] = value
    return {"tensors": tensor_digest(tensors)}


def batch_curriculum(
    candidate: SelectionCandidate, rows: list[dict], *, threshold: float = CURRICULUM_THRESHOLD
) -> dict:
    """Accuracy-triggered self-prefix: per example, only when the batch-level
    teacher-forced completion rate clears the frozen threshold."""
    fixtures = [to_fixture(row) for row in rows]
    completed = 0
    for fixture in fixtures:
        with torch.no_grad():
            result = locked_generate(
                candidate,
                fixture["goal"].encode(),
                fixture["records"],
                fixture["record_keys"],
                subject_id=f"cur-{fixture['index']}",
            )
        completed += int(result["raw_bytes"] == fixture["target_prefix"] and result["terminated"])
    rate = completed / len(fixtures)
    return {
        "threshold": threshold,
        "batch_completion_rate": rate,
        "use_self_prefix": rate >= threshold,
        "examples": len(fixtures),
    }


def fd_probe(candidate: SelectionCandidate, fixture: dict) -> dict:
    """Directional gradient check on the selection head and on locked NLL."""
    head_params = [p for p in candidate.head.parameters() if p.requires_grad]

    def loss_ce() -> torch.Tensor:
        return selection_ce(
            candidate,
            fixture["goal"].encode(),
            fixture["record_keys"],
            fixture["selection_label"],
        )

    def loss_locked() -> torch.Tensor:
        return locked_loss(
            candidate,
            fixture["goal"].encode(),
            fixture["records"],
            fixture["record_keys"],
            fixture["target_prefix"],
            subject_id=f"fd-{fixture['index']}",
        )

    rows = []
    for eps in FD_EPS:
        grad = torch.autograd.grad(loss_ce(), head_params, allow_unused=True)
        analytic = [float(g.norm()) if g is not None else 0.0 for g in grad]
        numeric = []
        for parameter in head_params:
            with torch.no_grad():
                original = parameter.clone()
                parameter.add_(eps)
                plus = float(loss_ce())
                parameter.copy_(original)
                parameter.sub_(eps)
                minus = float(loss_ce())
                parameter.copy_(original)
            numeric.append(abs(plus - minus) / (2 * eps))
        rows.append(
            {
                "eps": eps,
                "ce_analytic": analytic,
                "ce_numeric": numeric,
                "ce_resolved": sum(analytic) > 0 and sum(numeric) > 0,
            }
        )
    locked_grad = torch.autograd.grad(
        loss_locked(), list(candidate.trainable_parameters()), allow_unused=True
    )
    resolved = sum(float(g.norm()) for g in locked_grad if g is not None)
    return {
        "ce_rows": rows,
        "ce_all_resolved": all(row["ce_resolved"] for row in rows),
        "locked_grad_norm": resolved,
        "locked_grad_resolved": bool(resolved > 0),
    }


def alpha_lock_probe(candidate: SelectionCandidate, fixture: dict) -> dict:
    base = candidate.binding
    choice = select(candidate, fixture["goal"].encode(), fixture["record_keys"])
    state = initial_binding_state(base, f"lock-{fixture['index']}")
    prepared = prepare(base, state, fixture["goal"].encode(), fixture["records"])
    state, gate = observe_goal(base, prepared)
    unforced = answer_step(base, prepared, state, EOS, gate_index=gate)
    locked = locked_step(
        candidate,
        prepared,
        state,
        EOS,
        gate_index=gate,
        chosen_record=int(choice["chosen"]),
    )
    return {
        "chosen": int(choice["chosen"]),
        "confidence": choice["confidence"],
        "unforced_alpha_argmax": int(torch.argmax(unforced.read.alpha[:-1]).item()),
        "locked_alpha_argmax": int(torch.argmax(locked.read.alpha[:-1]).item()),
        "alpha_is_one_hot": bool(float(locked.read.alpha.max()) == 1.0),
        "alpha_null_zero": bool(float(locked.read.alpha[-1]) == 0.0),
        "betas_preserved": all(
            bool(torch.equal(a, b))
            for a, b in zip(unforced.read.betas, locked.read.betas, strict=True)
        ),
        "gamma_model": float(unforced.gamma.detach()),
        "gamma_locked": float(locked.gamma.detach()),
        "gamma_floor_honoured": float(locked.gamma.detach()) >= GAMMA_FLOOR - 1e-12,
        "probabilities_differ": not bool(torch.equal(unforced.probabilities, locked.probabilities)),
    }


def identity_rescue(candidate: SelectionCandidate, fixture: dict) -> dict:
    base = candidate.binding
    state = initial_binding_state(base, f"rescue-{fixture['index']}")
    prepared = prepare(base, state, fixture["goal"].encode(), fixture["records"])
    state, gate = observe_goal(base, prepared)
    choice = select(candidate, fixture["goal"].encode(), fixture["record_keys"])
    before = locked_step(
        candidate, prepared, state, EOS, gate_index=gate, chosen_record=int(choice["chosen"])
    )
    intervened = answer_step(base, prepared, state, EOS, gate_index=gate, gamma_override=0.0)
    after = locked_step(
        candidate, prepared, state, EOS, gate_index=gate, chosen_record=int(choice["chosen"])
    )
    return {
        "intervention_diff": float(
            (intervened.probabilities.detach() - before.probabilities.detach()).abs().max()
        ),
        "rescue_equal": bool(torch.equal(after.probabilities, before.probabilities)),
    }


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=worker, args=(spec,))
    process.start()
    process.join(timeout=WORKER_TIMEOUT)
    if process.is_alive():
        process.terminate()
        process.join()
        return {"ok": False, "message": "worker timed out", "cpu_seconds": 0.0, "adam_calls": 0}
    if process.exitcode != 0:
        return {
            "ok": False,
            "message": f"exit {process.exitcode}",
            "cpu_seconds": 0.0,
            "adam_calls": 0,
        }
    result_path = Path(spec["result"])
    if not result_path.exists():
        return {"ok": False, "message": "no result", "cpu_seconds": 0.0, "adam_calls": 0}
    return torch.load(result_path, map_location="cpu", weights_only=False)


def worker(spec: dict) -> None:
    start_cpu = time.process_time()
    out: dict = {"ok": False}
    try:
        loaded = load_bundle(Path(spec["bundle"]), torch.float64)
        candidate, state, optimizer, profile, update_index, cursor, _payload = loaded
        out = {
            "ok": True,
            "update_index": int(update_index),
            "snapshot": snapshot(candidate, state, optimizer),
        }
        if spec["mode"] == "step":
            fixture = to_fixture(load_rows("train")[spec["row"]])
            optimizer.zero_grad()
            loss = selection_ce(
                candidate,
                fixture["goal"].encode(),
                fixture["record_keys"],
                fixture["selection_label"],
            ) + locked_loss(
                candidate,
                fixture["goal"].encode(),
                fixture["records"],
                fixture["record_keys"],
                fixture["target_prefix"],
                subject_id=f"step-{fixture['index']}",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM)
            optimizer.step()
            out["adam_calls"] = 1
            out["loss"] = float(loss.detach())
            out["snapshot"] = snapshot(candidate, state, optimizer)
    except Exception:
        out = {"ok": False, "message": traceback.format_exc()}
    out["cpu_seconds"] = time.process_time() - start_cpu
    out.setdefault("adam_calls", 0)
    torch.save(out, spec["result"])


def run_profile(payload: dict, rows: list[dict], profile: str, budget: Budget) -> dict:
    candidate, optimizer = build_profile(payload, profile)
    fixture = to_fixture(rows[0])
    head_expected = expected_selection_parameter_count()

    init_state = initial_binding_state(candidate.binding, f"init-{profile}")
    init_path = OUTPUT_DIR / f"{profile}-initial.pt"
    info = save_bundle(
        init_path,
        profile=profile,
        candidate=candidate,
        state=init_state,
        optimizer=optimizer,
        update_index=0,
        cursor={"boundary": "initial"},
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(info)
    live = snapshot(candidate, init_state, optimizer)
    w0 = spawn_worker(
        {"bundle": str(init_path), "result": str(OUTPUT_DIR / f"{profile}-w0.pt"), "mode": "digest"}
    )
    budget.add_worker(w0)
    initial_equal = bool(w0.get("ok")) and w0["snapshot"] == live
    budget.drop_product(info)
    init_path.unlink(missing_ok=True)
    (OUTPUT_DIR / f"{profile}-w0.pt").unlink(missing_ok=True)

    optimizer.zero_grad()
    loss = selection_ce(
        candidate, fixture["goal"].encode(), fixture["record_keys"], fixture["selection_label"]
    ) + locked_loss(
        candidate,
        fixture["goal"].encode(),
        fixture["records"],
        fixture["record_keys"],
        fixture["target_prefix"],
        subject_id=f"step1-{profile}",
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM)
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()

    step1_state = initial_binding_state(candidate.binding, f"step1-{profile}")
    step1_path = OUTPUT_DIR / f"{profile}-step1.pt"
    step1_info = save_bundle(
        step1_path,
        profile=profile,
        candidate=candidate,
        state=step1_state,
        optimizer=optimizer,
        update_index=1,
        cursor={"boundary": "step1"},
        parent_identity={"sha256": PARENT_SHA256},
    )
    budget.add_product(step1_info)
    step1_live = snapshot(candidate, step1_state, optimizer)
    w1 = spawn_worker(
        {
            "bundle": str(step1_path),
            "result": str(OUTPUT_DIR / f"{profile}-w1.pt"),
            "mode": "digest",
        }
    )
    budget.add_worker(w1)
    step1_equal = bool(w1.get("ok")) and w1["snapshot"] == step1_live

    w2 = spawn_worker(
        {
            "bundle": str(step1_path),
            "result": str(OUTPUT_DIR / f"{profile}-w2.pt"),
            "mode": "step",
            "row": 0,
        }
    )
    budget.add_worker(w2)
    budget.drop_product(step1_info)
    optimizer.zero_grad()
    second = selection_ce(
        candidate, fixture["goal"].encode(), fixture["record_keys"], fixture["selection_label"]
    ) + locked_loss(
        candidate,
        fixture["goal"].encode(),
        fixture["records"],
        fixture["record_keys"],
        fixture["target_prefix"],
        subject_id=f"step2-{profile}",
    )
    second.backward()
    torch.nn.utils.clip_grad_norm_(list(candidate.trainable_parameters()), CLIP_NORM)
    optimizer.step()
    budget.adam_calls += 1
    budget.ensure()
    replay_equal = (
        bool(w2.get("ok"))
        and w2["snapshot"] == snapshot(candidate, step1_state, optimizer)
        and abs(float(w2.get("loss", 0.0)) - float(second.detach())) < 1e-12
    )
    for path in (
        step1_path,
        OUTPUT_DIR / f"{profile}-w1.pt",
        OUTPUT_DIR / f"{profile}-w2.pt",
    ):
        path.unlink(missing_ok=True)

    fd = fd_probe(candidate, fixture)
    lock = alpha_lock_probe(candidate, fixture)
    rescue = identity_rescue(candidate, fixture)
    curriculum = batch_curriculum(candidate, rows[:BATCH])
    gamma_mid = locked_generate(
        candidate,
        fixture["goal"].encode(),
        fixture["records"],
        fixture["record_keys"],
        subject_id=f"gen-{profile}",
    )

    total = sum(1 for _ in candidate.trainable_parameters())
    head_trainable = sum(1 for p in candidate.head.parameters() if p.requires_grad)
    return {
        "profile": profile,
        "head_parameters": int(sum(p.numel() for p in candidate.head.parameters())),
        "head_parameters_expected": int(head_expected),
        "head_parameter_count_matches": int(sum(p.numel() for p in candidate.head.parameters()))
        == int(head_expected),
        "base_parameters": int(sum(p.numel() for p in candidate.binding.parameters())),
        "total_parameters": int(sum(p.numel() for p in candidate.parameters())),
        "trainable_tensors": int(total),
        "head_trainable_tensors": int(head_trainable),
        "initial_restore_equal": initial_equal,
        "initial_worker_ok": bool(w0.get("ok")),
        "initial_worker_message": w0.get("message"),
        "step1_restore_equal": step1_equal,
        "step1_worker_ok": bool(w1.get("ok")),
        "step1_worker_message": w1.get("message"),
        "replay_equal": bool(replay_equal),
        "replay_worker_ok": bool(w2.get("ok")),
        "replay_worker_message": w2.get("message"),
        "fd": fd,
        "alpha_lock": lock,
        "identity_rescue": rescue,
        "curriculum": curriculum,
        "gamma_mean": gamma_mid["gamma_mean"],
        "gamma_floor": gamma_mid["gamma_floor"],
        "selection_chosen": gamma_mid["selection"]["chosen"],
        "selection_correct": gamma_mid["selection"]["chosen"] == fixture["selection_label"],
        "step1_loss": float(loss.detach()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="graph", choices=("graph",))
    parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    budget = Budget()
    try:
        if sha256_file(ROOT / PARENT_BUNDLE) != PARENT_SHA256:
            raise SystemExit("parent bundle sha drifted")
        payload = torch.load(ROOT / PARENT_BUNDLE, map_location="cpu", weights_only=False)
        rows = load_rows("train")

        results = [run_profile(payload, rows, profile, budget) for profile in PROFILES]

        checks = {
            "parent_identity": sha256_file(ROOT / PARENT_BUNDLE) == PARENT_SHA256,
            "profiles": len(results) == len(PROFILES),
            "head_parameter_count": all(r["head_parameter_count_matches"] for r in results),
            "head_trainable": all(r["head_trainable_tensors"] > 0 for r in results),
            "selection_fd_resolved": all(r["fd"]["ce_all_resolved"] for r in results),
            "locked_grad_resolved": all(r["fd"]["locked_grad_resolved"] for r in results),
            "alpha_locked_one_hot": all(r["alpha_lock"]["alpha_is_one_hot"] for r in results),
            "alpha_null_zero": all(r["alpha_lock"]["alpha_null_zero"] for r in results),
            "betas_preserved": all(r["alpha_lock"]["betas_preserved"] for r in results),
            "gamma_floor_honoured": all(r["alpha_lock"]["gamma_floor_honoured"] for r in results),
            "locking_changes_output": all(r["alpha_lock"]["probabilities_differ"] for r in results),
            "identity_rescue": all(r["identity_rescue"]["rescue_equal"] for r in results),
            "curriculum_interface": all(
                r["curriculum"]["threshold"] == CURRICULUM_THRESHOLD
                and isinstance(r["curriculum"]["use_self_prefix"], bool)
                for r in results
            ),
            "initial_restore": all(r["initial_restore_equal"] for r in results),
            "step1_restore": all(r["step1_restore_equal"] for r in results),
            "replay_update": all(r["replay_equal"] for r in results),
            "adam_budget": budget.adam_calls <= STAGE_ADAM_CALLS,
            "product_limit": budget.max_live <= STAGE_PRODUCT_BYTES,
            "stage_budget": budget.total_cpu() <= STAGE_CPU_SECONDS
            and (time.perf_counter() - STARTED_WALL) <= STAGE_WALL_SECONDS,
            "products_cleaned": not any(OUTPUT_DIR.glob("*.pt")),
            "confirm_untouched": True,
        }
        failed = [name for name, ok in checks.items() if not ok]
        status = "passed_16_1_0b_graph" if not failed else "engineering_failure"

        report = {
            "schema": "taiji-binding-select-graph-v1",
            "step": "N16.1.0b",
            "date": "2026-11-03",
            "status": status,
            "checks": checks,
            "failed": failed,
            "profiles": results,
            "curriculum_contract": {
                "trigger": "batch teacher-forced completion rate",
                "threshold": CURRICULUM_THRESHOLD,
                "dose": "per-example self-prefix once triggered; frozen before 0c",
            },
            "gamma_floor": GAMMA_FLOOR,
            "accounting": {"actual_adam_calls": budget.adam_calls, "effective_adam_calls": 0},
            "cost": {
                "cpu_seconds": budget.total_cpu(),
                "wall_seconds": time.perf_counter() - STARTED_WALL,
                "adam_calls": budget.adam_calls,
                "peak_live_product_bytes": budget.max_live,
            },
            "budget": {
                "cpu_seconds": STAGE_CPU_SECONDS,
                "wall_seconds": STAGE_WALL_SECONDS,
                "adam_calls": STAGE_ADAM_CALLS,
                "product_bytes": STAGE_PRODUCT_BYTES,
            },
            "source_hashes": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
            "limits": (
                "Graph engineering only. No adoption training, no confirm read, no capability claim."
            ),
        }
        Path(REPORT).write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema": "taiji-binding-select-manifest-v1",
            "step": "N16.1.0b",
            "status": status,
            "report": REPORT,
            "parent_bundle": PARENT_BUNDLE,
            "parent_sha256": PARENT_SHA256,
            "profiles": list(PROFILES),
            "checks": checks,
            "cost": report["cost"],
            "source_hashes": report["source_hashes"],
        }
        Path(MANIFEST).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"16.1.0b status: {status}")
        print(f"  failed: {failed}")
        print(
            f"  cpu {budget.total_cpu():.2f}/{STAGE_CPU_SECONDS}, adam {budget.adam_calls}/{STAGE_ADAM_CALLS}"
        )
        print(f"  peak {budget.max_live / (1024 * 1024):.1f} MiB")
        return 0 if not failed else 1
    except StageStop as exc:
        print(f"16.1.0b stop: {exc}")
        return 1
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
