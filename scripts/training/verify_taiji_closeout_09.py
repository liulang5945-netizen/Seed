"""09.8 closeout: six artefacts, three consistency checks, one capability card.

Produces the subject bundle (kept under output/, not in git), the structure and
inheritance maps, a real activity trace, two raw outputs, and the capability and
resource card for every 09 sub-step. Nothing here claims a capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import scripts.training.verify_taiji_birth_direction as rule  # noqa: E402
import scripts.training.verify_taiji_living_fragment_pricing_r028 as pricing  # noqa: E402
import scripts.training.verify_taiji_netbenefit_second as second  # noqa: E402
from taiji import birth_direction as bd  # noqa: E402
from taiji import growth_credit as gc  # noqa: E402
from taiji import native_development as development  # noqa: E402
from taiji.activity_learning import ActivityLearner, train_activity_episode  # noqa: E402
from taiji.living_core import LivingCoreReference  # noqa: E402
from taiji.native_activity import ActivityEnvironment, run_round  # noqa: E402
from taiji.native_activity_fixture import ArrivalSource  # noqa: E402
from taiji.state_migration import (  # noqa: E402
    load_subject,
    resume_career,
    resume_subject,
    run_worker_job,
    save_subject,
)

SEED = 20261101
PARTS = 5
UNITS_PER_GROUP = 3
STEPS = 8
LR = 1e-5
ACTIVITY_ROUNDS = 8
ACTIVITY_TRAIN_ROUNDS = 256
BUDGET_CPU_SECONDS = 900
BUDGET_WALL_SECONDS = 1800
CAPABILITY_WORDS = ("能力", "学会", "掌握", "L3", "capability", "autonom")
SUB_STEPS = (
    ("09.6", "reports/taiji_core_r028_life_fragment_pricing_20261101.json"),
    ("09.6.1", "reports/taiji_core_r029_engineering_gaps_20261101.json"),
    ("09.6.2", "reports/taiji_core_fixture_arrivals_v2_20261101.json"),
    ("09.6.3", "reports/taiji_core_birth_inertness_20261101.json"),
    ("09.6.3b", "reports/taiji_core_direction_oracle_20261101.json"),
    ("09.6.4", "reports/taiji_core_birth_direction_20261101.json"),
    ("09.6.5", "reports/taiji_core_birth_direction_abs_20261101.json"),
    ("09.6.6", "reports/taiji_core_birth_direction_parts_20261101.json"),
    ("09.6.7", "reports/taiji_core_birth_direction_surface_20261101.json"),
    ("09.6.8", "reports/taiji_core_birth_netbenefit_20261101.json"),
    ("09.6.9", "reports/taiji_core_netbenefit_second_20261101.json"),
    ("09.7", "reports/taiji_core_fragment_09_7_20261101.json"),
    ("09.7b", "reports/taiji_core_fragment_horizon_20261101.json"),
)
CLAIMS = {
    "09.6": ("费用与单价已按声明夹具实测", "不签任务表现、保持或生长收益"),
    "09.6.1": ("五项工程缺口已实现并各自过门", "lesion 贡献判负，不签结构有效"),
    "09.6.2": ("活动夹具具备动态范围，生长网格六格全发布", "不签自主性，生长面四次否证"),
    "09.6.3": ("新生支路确实被训练但仍无可测贡献", "不签结构有效"),
    "09.6.3b": ("方向可见评估集时新生份额上升", "神谕是可达性上界，不是可用方法"),
    "09.6.4": ("份额判据被自身随机对照证伪", "该轮不构成对 H1 的支持"),
    "09.6.5": ("种子轴无效，结论作废", "该状态不得引用"),
    "09.6.6": ("窗口数轴上 2/3 格过绝对门", "单轴复现，不外推"),
    "09.6.7": ("15 格中 8 格过门，覆盖 4 个 parts", "薄平台，parts=4 三档全败"),
    "09.6.8": ("出生比不出生多改善 1.739e-4，代价 11.50 CPU 秒", "单一确定性主体、合成夹具"),
    "09.6.9": ("第二条评估生涯上净收益复现", "两条均为合成夹具"),
    "09.7": ("32 步净收益升到 7.187e-4／8.147e-4", "两臂同时破保持门，卡在训练预算"),
    "09.7b": ("保持门内最长声明检查点为 8 步", "不是机制的可持续上限"),
}


class Budget(Exception):
    pass


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(relative: str) -> str:
    return digest_bytes((ROOT / relative).read_bytes())


def inheritance_map(parent, child, split: int) -> dict:
    parent_state = dict(parent.named_parameters())
    child_state = dict(child.named_parameters())
    rows = []
    equal = 0
    for name, parameter in parent_state.items():
        counterpart = child_state[name]
        if name == "R":
            left = counterpart.detach()[:, :split, :]
            right = parameter.detach()
        elif name == "P":
            left = counterpart.detach()[:, :, :split]
            right = parameter.detach()
        else:
            left, right = counterpart.detach(), parameter.detach()
        same = bool(torch.equal(left, right))
        equal += int(same)
        rows.append(
            {
                "name": name,
                "bitwise_equal": same,
                "max_abs_diff": (
                    float((left - right).abs().max()) if left.shape == right.shape else None
                ),
            }
        )
    return {"equal": equal, "total": len(rows), "parameters": rows}


def spawn_worker(spec: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=run_worker_job, args=(spec,))
    process.start()
    process.join(timeout=600)
    if process.exitcode != 0:
        raise RuntimeError(f"worker exitcode={process.exitcode}")
    return torch.load(spec["out"], weights_only=True)


def raw_output(model, journal, provider) -> dict:
    with torch.no_grad():
        _, steps = resume_career(
            model,
            model.initial_state(journal.subject_id),
            journal,
            0,
            provider,
            retain_outputs=False,
        )
    first = next((probabilities for probabilities, byte in steps if byte is not None), None)
    if first is None:
        return {"top1_byte": None, "top3_probabilities": []}
    values, indices = torch.topk(first, 3)
    return {"top1_byte": int(indices[0]), "top3_probabilities": [float(value) for value in values]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "output" / "taiji_core_09")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("09.8 card is write-once")

    torch.set_num_threads(1)
    total_cpu, total_wall = time.process_time(), time.perf_counter()

    def guard() -> None:
        if time.process_time() - total_cpu > BUDGET_CPU_SECONDS:
            raise Budget("cpu budget exhausted")
        if time.perf_counter() - total_wall > BUDGET_WALL_SECONDS:
            raise Budget("wall budget exhausted")

    torch.manual_seed(SEED)
    parent = LivingCoreReference(pricing.PRINCIPLE).double()
    guard()
    directions = bd.consensus_directions(
        deepcopy(parent), pricing.LONG, parts=PARTS, count=UNITS_PER_GROUP
    )
    child, _plan = gc.birth(deepcopy(parent), directions, rule._credits(parent))
    split = child.growth_parent_units
    inheritance = inheritance_map(parent, child, split)
    structure = {
        "parent_parameters": sum(parameter.numel() for parameter in parent.parameters()),
        "child_parameters": sum(parameter.numel() for parameter in child.parameters()),
        "growth_parent_units": split,
        "newborn_units_per_group": int(child.R.shape[1]) - int(split),
        "units_total": (int(child.R.shape[1]) - int(split)) * parent.config.groups,
    }

    guard()
    provider = rule.open_gates(bd.RULE_NAME)
    optimizer = torch.optim.Adam(child.parameters(), lr=LR)
    losses, train_cpu, train_wall = pricing.timed(
        lambda: development.declared_feedback_updates(
            child, pricing.LONG, optimizer, provider, steps=STEPS
        )
    )

    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = args.bundle_dir / "subject_bundle.pt"
    with torch.no_grad():
        state, _ = resume_career(
            child, child.initial_state("taiji-core-09"), pricing.SHORT, 0, provider, limit=1
        )
    guard()
    bundle_sha, save_cpu, _save_wall = pricing.timed(
        lambda: save_subject(
            bundle_path,
            child,
            state,
            career=pricing.SHORT,
            journal_cursor=1,
            optimizer=optimizer,
            eligibility=torch.zeros(pricing.PRINCIPLE.groups, dtype=torch.float64),
            journal_entries=[{"seq": 0, "kind": "closeout_seed"}],
            pending=None,
            growth_gates=[dict(provider(0))],
        )
    )

    restored_model, restored_state, _ = resume_subject(load_subject(bundle_path))
    with torch.no_grad():
        local_state, local_steps = resume_career(
            restored_model, restored_state, pricing.SHORT, 1, provider, limit=1
        )
    guard()
    with tempfile.TemporaryDirectory() as directory:
        worker = spawn_worker(
            {
                "bundle": str(bundle_path),
                "mode": "microstep",
                "out": str(Path(directory) / "worker.pt"),
                "out_bundle": None,
            }
        )
    restore_equal = bool(
        torch.equal(worker["h"], local_state.h)
        and torch.equal(worker["probabilities"], local_steps[0][0].probabilities)
    )

    guard()
    learner = ActivityLearner.build(pricing.PRINCIPLE.width)
    train_activity_episode(
        child,
        state,
        ActivityEnvironment(store=pricing.RECORDS, goal=pricing.CHINESE_GOAL, now=0),
        learner,
        pricing.SHORT,
        pricing.RETENTION,
        rounds=ACTIVITY_TRAIN_ROUNDS,
        journal=[],
        last_event=torch.zeros(pricing.PRINCIPLE.width, dtype=torch.float64),
        max_exposures=ACTIVITY_TRAIN_ROUNDS,
        arrivals=ArrivalSource(per_round=True),
        provider=provider,
    )
    policy = learner.policy
    env = ActivityEnvironment(store=pricing.RECORDS, goal=pricing.CHINESE_GOAL, now=0)
    arrivals = ArrivalSource(per_round=True)
    last_event = torch.zeros(pricing.PRINCIPLE.width, dtype=torch.float64)
    trace = []
    with torch.no_grad():
        for _ in range(ACTIVITY_ROUNDS):
            state, env, entry, last_event, _ = run_round(
                child,
                state,
                env,
                policy,
                [],
                last_event=last_event,
                prediction_error_value=0.0,
                arrivals=arrivals,
                gate_provider=provider,
            )
            trace.append(
                {
                    "activity": entry["activity"],
                    "chosen_by": entry["chosen_by"],
                    "step_delta": entry["step_delta"],
                    "external_trigger": entry["external_trigger"],
                    "triggers": entry["triggers"],
                }
            )

    outputs = {
        "holdout": raw_output(child, pricing.HOLDOUT, provider),
        "second_holdout": raw_output(child, second.SECOND_HOLDOUT, provider),
    }

    sub_steps = []
    for step, relative in SUB_STEPS:
        path = ROOT / relative
        record = {"step": step, "report": relative, "exists": path.exists()}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            record["status"] = data.get("status") or data.get("verdict") or "recorded"
            record["sha256"] = digest_bytes(path.read_bytes())
        record["claimable"], record["not_claimable"] = CLAIMS[step]
        sub_steps.append(record)

    claim_text = " ".join(f"{row['claimable']} {row['not_claimable']}" for row in sub_steps)
    forbidden = [word for word in CAPABILITY_WORDS if word in claim_text]
    # Under the per-round supply every round can legally interact and pay, so a
    # competent policy is expected to interact every round; demanding a wait would
    # contradict the fixture rather than test the subject.
    trace_ok = len(trace) == ACTIVITY_ROUNDS and any(row["activity"] == "interact" for row in trace)
    trace_complete = all(row["exists"] for row in sub_steps)

    cpu_seconds = time.process_time() - total_cpu
    wall_seconds = time.perf_counter() - total_wall
    if cpu_seconds > BUDGET_CPU_SECONDS or wall_seconds > BUDGET_WALL_SECONDS:
        status = "budget_stop"
    elif not trace_complete:
        status = "trace_incomplete"
    elif forbidden:
        status = "capability_claim_forbidden"
    elif inheritance["equal"] != inheritance["total"]:
        status = "inheritance_mismatch"
    elif not restore_equal:
        status = "restore_mismatch"
    elif not trace_ok:
        status = "degenerate_activity_trace"
    else:
        status = "closeout_passed_engineering_only"

    checks = [
        {
            "id": "all_sub_step_reports_present",
            "passed": trace_complete,
            "observed": len(sub_steps),
        },
        {"id": "restore_bitwise", "passed": restore_equal, "observed": restore_equal},
        {
            "id": "inheritance_bitwise",
            "passed": inheritance["equal"] == inheritance["total"],
            "observed": [inheritance["equal"], inheritance["total"]],
        },
        {"id": "activity_trace_non_degenerate", "passed": trace_ok, "observed": trace},
        {"id": "no_capability_claim", "passed": not forbidden, "observed": forbidden},
        {
            "id": "budget_not_exceeded",
            "passed": cpu_seconds <= BUDGET_CPU_SECONDS and wall_seconds <= BUDGET_WALL_SECONDS,
            "observed": [cpu_seconds, wall_seconds],
        },
    ]
    card = {
        "schema": "taiji-09-closeout-v1",
        "step": "09.8",
        "date": "2026-11-01",
        "status": status,
        "identity": {
            "config": asdict(pricing.PRINCIPLE),
            "parent_model_digest": digest_bytes(
                b"".join(
                    name.encode("utf-8") + parameter.detach().cpu().contiguous().numpy().tobytes()
                    for name, parameter in sorted(parent.named_parameters())
                )
            ),
            "direction_rule": {
                "name": bd.RULE_NAME,
                "parts": PARTS,
                "units_per_group": UNITS_PER_GROUP,
            },
            "declared_updates": STEPS,
            "lr": LR,
        },
        "bundle": {
            "path": str(bundle_path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": bundle_sha,
            "bytes": bundle_path.stat().st_size,
            "in_git": False,
            "reproduce": "python scripts/training/verify_taiji_closeout_09.py --output <path>",
        },
        "structure": structure,
        "inheritance": inheritance,
        "restore": {"spawn_bitwise_equal": restore_equal},
        "training": {"losses": losses, "cpu_seconds": train_cpu, "wall_seconds": train_wall},
        "activity_trace": trace,
        "activity_policy": {
            "source": "trained_in_this_closeout",
            "train_rounds": ACTIVITY_TRAIN_ROUNDS,
            "fixture": "per_round arrivals (owner-approved activity face)",
            "claim": "sequence recorded; under per-round supply interact is legal and paid every round",
        },
        "raw_outputs": outputs,
        "sub_steps": sub_steps,
        "checks": checks,
        "training_admitted": False,
        "task_capability_claimed": False,
        "limits": (
            "Engineering closeout of step 09 only. The subject is a synthetic-fixture candidate "
            "on one deterministic reference graph; the card records what each sub-step measured "
            "and what it cannot claim. No capability, no autonomy and no training admission."
        ),
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "budget": {"cpu_seconds": BUDGET_CPU_SECONDS, "wall_seconds": BUDGET_WALL_SECONDS},
        "source_hashes": {
            relative: file_sha(relative)
            for relative in (
                "taiji/living_core.py",
                "taiji/growth_credit.py",
                "taiji/native_development.py",
                "taiji/native_activity.py",
                "taiji/native_activity_fixture.py",
                "taiji/birth_direction.py",
                "taiji/state_migration.py",
                "scripts/training/verify_taiji_closeout_09.py",
            )
        },
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(card, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    print(
        json.dumps(
            {
                "status": status,
                "bundle_bytes": card["bundle"]["bytes"],
                "inheritance": [inheritance["equal"], inheritance["total"]],
                "restore_equal": restore_equal,
                "activities": [row["activity"] for row in trace],
                "cpu_seconds": round(cpu_seconds, 2),
                "failed": [row["id"] for row in checks if not row["passed"]],
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
