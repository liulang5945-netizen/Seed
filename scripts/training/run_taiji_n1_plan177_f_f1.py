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
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.evaluate_taiji_n1_plan177_f_f1_holdout import (  # noqa: E402
    _replay,
    _store_from_parse,
)
from scripts.training.run_taiji_n1_plan177_f_f0b import (  # noqa: E402
    ARMS,
    BATCH,
    _Arm,
    _rel,
    _sha256_file,
    load_trainer,
)

RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f_f1_20261029.json"
F1_UPDATES = 1280
CHECKPOINT_EVERY = 256
RUN_CPU_CAP = 10100.0
TOTAL_CPU_CAP = 121500.0
SEEDS = (0, 1, 2)


def build_cases(trainer: Any) -> list[dict[str, Any]]:
    """训练面：全部 train episode 里**解析出实体且有当前状态**的查询轮。

    F1 只用了每家族 `:0` 的 32 条（其中 16 条可解析），训练侧 answer_exact 只有 2.3%——
    那是训练覆盖不足，不是机制结论。这里改成逐轮收集，特征取 store 选出的当前值
    （与评分面同源：模型只能用它自己解析到的状态）。
    """

    from taiji.joint_state_consumption import PersistentContentStoreV1
    from taiji.joint_state_consumption_course import E1_ATTRIBUTE, build_train_episodes

    cases: list[dict[str, Any]] = []
    for episode in build_train_episodes():
        rows = _replay(trainer, episode)
        store = PersistentContentStoreV1(())
        for index, row in enumerate(rows):
            store = _store_from_parse(store, row["spans"], turn_index=index, revision=index)
            if row["act"] != "query":
                continue
            entity = row["spans"].get("entity")
            current = store.current(entity, E1_ATTRIBUTE) if entity else None
            if current is None:
                continue
            cases.append(
                {
                    "case_id": f"{episode.episode_id}:{index}",
                    "joint": row["joint"],
                    "target": torch.tensor(
                        list(row["turn"].answer_target.target_bytes), dtype=torch.long
                    ),
                    "value": current.value,
                }
            )
    return cases


def answer_loss(arm: _Arm, trainer: Any, case: dict[str, Any]) -> torch.Tensor:
    """教师强制整句答案：每步都加同一 bias，损失由答案字节逐步 CE 组成。"""

    joint = case["joint"]
    target = case["target"]
    bias = arm.bias(arm.feature({"value": case["value"]}))
    state = joint.language_state
    source = joint.input_bindings[-1].neural_source_id
    cache = trainer.model._joint_evidence_cache(joint, selected_source_ids=(source,))
    logits = []
    for symbol in target[:-1]:
        step_logits, hidden = trainer.model.next_logits(state, cache=cache)
        logits.append(step_logits + bias)
        state = trainer.model.observe_response(state, int(symbol), hidden)
    return F.cross_entropy(torch.stack(logits), target[1:])


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


def _update(arm: _Arm, trainer: Any, batch: list[dict[str, Any]]) -> float:
    losses = [answer_loss(arm, trainer, case) for case in batch]
    loss = torch.stack(losses).mean()
    arm.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    arm.optimizer.step()
    arm.cursor += 1
    return float(loss.detach())


CHILD_SCRIPT = chr(10).join(
    [
        "import json, sys, torch",
        "from taiji.internalization import content_digest",
        "from scripts.training.run_taiji_n1_plan177_f_f0b import _Arm",
        "arm = _Arm.from_checkpoint(torch.load(sys.argv[1], map_location='cpu', weights_only=False))",
        "print(json.dumps({",
        "    'parameters': arm.parameters_digest(),",
        "    'optimizer': content_digest(arm.optimizer.state_dict()),",
        "    'cursor': arm.cursor,",
        "}))",
    ]
)


def child_digest(checkpoint: Path) -> dict[str, Any]:
    """新进程加载端点，回传参数/优化器/cursor 摘要——恢复门的最薄一支。"""

    script = CHILD_SCRIPT
    completed = subprocess.run(
        [sys.executable, "-c", script, str(checkpoint)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1"),
        timeout=600,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"F1 endpoint restore failed: {completed.stderr[-600:]}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def run_one(
    arm_name: str,
    seed: int,
    trainer: Any,
    cases: list[dict[str, Any]],
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
        _update(arm, trainer, [cases[(offset + index) % len(cases)] for index in range(BATCH)])
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
    from taiji.internalization import content_digest as _cd

    parent_digest = {
        "parameters": arm.parameters_digest(),
        "optimizer": _cd(arm.optimizer.state_dict()),
        "cursor": arm.cursor,
    }
    restored_digest = child_digest(endpoint)
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
            "endpoint_restore_ok": restored_digest == parent_digest,
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
    cases = build_cases(trainer)
    if not cases:
        raise RuntimeError("no parseable train query turns; F1b cannot train")
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
            arm_name,
            seed,
            trainer,
            cases,
            run_root,
            updates=args.updates,
            threads=args.threads,
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
            "train_cases": len(cases),
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
