"""N177 F1：四臂×三seed、1,280 updates/run 的固定训练（可续跑）。

候选按 owner 2026-10-29 裁定：producer/V4 冻结，只训 value encoder+bias（F0b 同一实现）。
每一步都重跑冻结链（producer→V4 query-only next-bytes→bias），所以单价与 F0b 实测同口径。

资源：整包上限 121,500 process-CPU 秒（owner 批准）；每 run 上限 10,100 CPU 秒；
每 256 步写耐久 checkpoint，端点另存；崩溃/暂停后从 `latest.pt` 续跑。
不读 query/holdout/sealed；不按中途表现调阈值或加轮次。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.run_taiji_n1_plan177_f_f0b import (  # noqa: E402
    ARMS,
    BATCH,
    _Arm,
    _rel,
    _sha256_file,
    load_trainer,
    train_items,
)

RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f_f1_20261029.json"
F1_UPDATES = 1280
CHECKPOINT_EVERY = 256
RUN_CPU_CAP = 10100.0
TOTAL_CPU_CAP = 121500.0
SEEDS = (0, 1, 2)


def _run_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run_one(
    arm_name: str,
    seed: int,
    items: list[dict[str, Any]],
    trainer: Any,
    run_root: Path,
    *,
    updates: int,
    threads: int,
) -> dict[str, Any]:
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "run_state.json"
    state = _run_state(state_path) or {
        "arm": arm_name,
        "seed": seed,
        "status": "starting",
        "updates": 0,
        "process_cpu_seconds": 0.0,
        "wall_seconds": 0.0,
    }
    latest = run_root / "latest.pt"
    if latest.is_file():
        arm = _Arm.from_checkpoint(torch.load(latest, map_location="cpu", weights_only=False))
        state["resumes"] = int(state.get("resumes", 0)) + 1
        state["status"] = "resumed"
    else:
        arm = _Arm(arm=arm_name, seed=seed)
        torch.save(arm.checkpoint(), run_root / "zero.pt")
        state["zero_sha256"] = _sha256_file(run_root / "zero.pt")

    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    state["status"] = "running"
    state["threads"] = threads
    _write_json(state_path, state)
    while arm.cursor < updates:
        if time.process_time() - cpu_before + float(state["process_cpu_seconds"]) >= RUN_CPU_CAP:
            state["status"] = "run_cpu_cap"
            break
        offset = BATCH * arm.cursor
        arm.step(
            trainer,
            [items[(offset + index) % len(items)] for index in range(BATCH)],
        )
        if arm.cursor % CHECKPOINT_EVERY == 0:
            torch.save(arm.checkpoint(), latest)
            _write_json(
                state_path,
                {
                    **state,
                    "updates": arm.cursor,
                    "durable_cursor": arm.cursor,
                    "process_cpu_seconds": float(state["process_cpu_seconds"])
                    + time.process_time()
                    - cpu_before,
                },
            )
    if arm.cursor < updates and state["status"] != "run_cpu_cap":
        state["status"] = "incomplete"
    torch.save(arm.checkpoint(), latest)
    endpoint = run_root / f"endpoint_update_{arm.cursor:04d}.pt"
    torch.save(arm.checkpoint(), endpoint)
    cpu_used = float(state["process_cpu_seconds"]) + time.process_time() - cpu_before
    wall_used = float(state["wall_seconds"]) + time.perf_counter() - wall_before
    state.update(
        {
            "updates": arm.cursor,
            "durable_cursor": arm.cursor,
            "process_cpu_seconds": cpu_used,
            "wall_seconds": wall_used,
            "endpoint": _rel(endpoint),
            "endpoint_sha256": _sha256_file(endpoint),
            "completed": arm.cursor >= updates,
        }
    )
    if state["completed"]:
        state["status"] = "completed"
    _write_json(state_path, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 F1 固定训练（可续跑）")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--updates", type=int, default=F1_UPDATES)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", default=None, help="只跑 arm__seed，例如 copy_only__seed_0")
    parser.add_argument("--max-runs", type=int, default=12)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    args.run_root.mkdir(parents=True, exist_ok=True)
    lock = args.run_root / "run_root.lock"
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"F1 run root already locked: {lock}") from error
    os.write(handle, json.dumps({"pid": os.getpid(), "started": time.time()}).encode())
    os.close(handle)

    trainer = load_trainer()
    items = train_items()
    planned = [(seed, arm) for seed in SEEDS for arm in ARMS]
    if args.only:
        name, seed_text = args.only.rsplit("__seed_", 1)
        planned = [(int(seed_text), name)]
    planned = planned[: args.max_runs]

    runs = []
    total_cpu = 0.0
    stop_reason = "completed_planned_runs"
    for seed, arm_name in planned:
        if total_cpu >= TOTAL_CPU_CAP:
            stop_reason = "total_cpu_cap"
            break
        run_root = args.run_root / f"{arm_name}__seed_{seed}"
        state = run_one(
            arm_name, seed, items, trainer, run_root, updates=args.updates, threads=args.threads
        )
        total_cpu += float(state["process_cpu_seconds"])
        runs.append(state)
        print(
            f"{arm_name}__seed_{seed}: updates={state['updates']} "
            f"cpu={state['process_cpu_seconds']:.1f} status={state['status']}",
            flush=True,
        )
        if not state["completed"]:
            stop_reason = f"{arm_name}__seed_{seed}:{state['status']}"
            break
    lock.unlink(missing_ok=True)
    _write_json(
        args.report,
        {
            "format": "taiji-n177-f-f1-v1",
            "date": "2026-10-29",
            "scope": "4 arms x 3 seeds; V4/producer frozen; encoder+bias trained; 1,280 updates",
            "updates": args.updates,
            "threads": args.threads,
            "total_cpu_cap": TOTAL_CPU_CAP,
            "run_cpu_cap": RUN_CPU_CAP,
            "stop_reason": stop_reason,
            "runs": runs,
            "total_process_cpu_seconds": total_cpu,
        },
    )
    print(f"stop_reason={stop_reason} total_cpu={total_cpu:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
