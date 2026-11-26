"""N177.6 E0：15 个终点的 fresh-process parse/feature/answer 回执（只读）。

为什么要有它：评价器在**同一进程**里读 15 个终点；"换一个进程还能复原同样的 parse/feature/answer"
从未逐臂记录过。E0 要的正是这份回执，并显式披露哪一臂绑定的实现 revision 与当前工作树不同。

做法（不重抄评分链）：直接复用评价器的 `_score_split`，输入固定在**训练集第一条** episode
（绝不读 holdout/query），先 fresh load 再评分一次，得到一份可哈希的逐轮读数。

两种执行面：
* `--arm <id>`（内部用）：在**新进程**里对一臂评分，把指纹打到 stdout；
* 默认：主进程对每臂评分，并对每臂再起一个真子进程复核，逐臂比较两个指纹。
子进程显式给 `PYTHONPATH` = 仓根（本机没有 `pip install -e .`，不给就会 `No module named 'taiji'`）。

绑定的实现 revision 取自 `metrics.jsonl` 里该臂**最后一条 update 事件**自带的
`implementation_revision`——那是它真正跑完时用的代码，不是今天的 HEAD。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_t1_course import build_joint_t1_split  # noqa: E402
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1  # noqa: E402
from taiji.joint_t1_trainer import JointT1TrainerV1  # noqa: E402

RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027"
TRAIN_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
REPORT = ROOT / "reports/taiji_n1_plan177_6_t1_endpoint_receipts_20261029.json"


def bound_revisions(run_root: Path) -> dict[str, str]:
    """每臂最后一条 update 事件自带的 revision＝该终点真正用过的代码身份。"""

    revisions: dict[str, str] = {}
    with (run_root / "metrics.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") == "update":
                revisions[event["arm_id"]] = event.get("implementation_revision") or ""
    return revisions


def endpoints(run_root: Path) -> dict[str, dict[str, Any]]:
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    return dict(state.get("latest_checkpoint_by_arm", {}))


_FIXED: tuple[Any, JointT1TraceCacheV1] | None = None


def fixed_input(cache_path: Path = TRAIN_CACHE) -> tuple[Any, JointT1TraceCacheV1]:
    """固定输入：训练集第一条（只读 train，绝不动 holdout/query）。

    课程与 590 MB 训练缓存**进程内只建一次**：逐臂重建会把 15 次解序列化算进读数，
    既慢又会让人误以为是评分本身贵。
    """

    global _FIXED
    if _FIXED is None:
        episodes = build_joint_t1_split("train")
        cache = JointT1TraceCacheV1.from_payload(
            torch.load(cache_path, map_location="cpu", weights_only=False)
        )
        _FIXED = (episodes[0], cache)
    return _FIXED


def _feed(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        hasher.update(b"T")
        hasher.update(str(tensor.dtype).encode())
        hasher.update(repr(tuple(tensor.shape)).encode())
        hasher.update(b"\n" if tensor.numel() == 0 else tensor.numpy().tobytes())
    elif isinstance(value, dict):
        hasher.update(b"D")
        for key in sorted(value, key=str):
            hasher.update(str(key).encode())
            _feed(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(b"L")
        for item in value:
            _feed(hasher, item)
    else:
        hasher.update(b"S")
        hasher.update(repr(value).encode())


def state_digest(*parts: Any) -> str:
    """权重/缓冲区/优化器状态的按字节摘要（浮点逐位，不含 repr 截断）。"""

    hasher = hashlib.sha256()
    for part in parts:
        _feed(hasher, part)
    return hasher.hexdigest()


def isolation_snapshot(trainer: JointT1TrainerV1) -> dict[str, Any]:
    """评分前后的 `(参数字节, 优化器状态, cursor)`——评分只许读，不许写。"""

    producer_state = (
        trainer.producer.state_dict() if hasattr(trainer.producer, "state_dict") else {}
    )
    return {
        "update_cursor": trainer.update_cursor,
        "producer_parameter_revision": trainer.producer.parameter_revision,
        "language_parameter_revision": trainer.model.parameter_revision,
        "model_state": state_digest(trainer.model.state_dict()),
        "producer_state": state_digest(producer_state),
        "optimizer_state": state_digest(
            trainer.producer_optimizer.state_dict(),
            trainer.language_trainer.optimizer.state_dict(),
        ),
    }


def fingerprint(
    arm: str, endpoint: Path, episode: Any, cache: JointT1TraceCacheV1
) -> dict[str, Any]:
    from scripts.training import evaluate_taiji_n1_plan177_6_t1_pilot as scorer

    payload = torch.load(endpoint, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=cache)
    before = isolation_snapshot(trainer)
    scores = scorer._score_split(
        trainer=trainer,
        episodes=(episode,),
        cache=cache,
        state_writes=bool(trainer.policy["cognitive_state_writes"]),
        budget_check=None,
    )
    after = isolation_snapshot(trainer)
    return {
        "arm": arm,
        "mode": trainer.mode,
        "seed": trainer.seed,
        "update_cursor": trainer.update_cursor,
        "state_writes": bool(trainer.policy["cognitive_state_writes"]),
        "episode_id": episode.episode_id,
        "scores": scores,
        "digest": content_digest(scores),
        "scoring_isolation": {
            "before": before,
            "after": after,
            "unchanged": before == after,
        },
    }


def run_one(arm: str, run_root: Path, cache_path: Path = TRAIN_CACHE) -> dict[str, Any]:
    endpoint = Path(endpoints(run_root)[arm]["path"])
    episode, cache = fixed_input(cache_path)
    return fingerprint(arm, endpoint, episode, cache)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177.6 E0 终点回执（只读）")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--arm", default=None, help="内部用：只评这一臂并打印指纹")
    parser.add_argument(
        "--all-fingerprints",
        action="store_true",
        help="内部用：在同一个新进程里评完全部臂并打印 {arm: digest}（缓存只加载一次）",
    )
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=TRAIN_CACHE,
        help="训练缓存（历史检出里用它自己的代码、但缓存仍在主树，必须显式给绝对路径）",
    )
    parser.add_argument("--out-report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)

    if args.arm is not None:
        result = run_one(args.arm, args.run_root, args.train_cache)
        sys.stdout.write(json.dumps({"arm": args.arm, "digest": result["digest"]}) + "\n")
        return 0

    if args.all_fingerprints:
        digests = {
            arm: run_one(arm, args.run_root, args.train_cache)["digest"]
            for arm in sorted(endpoints(args.run_root))
        }
        sys.stdout.write(json.dumps({"digests": digests}) + "\n")
        return 0

    revisions = bound_revisions(args.run_root)
    arms = sorted(endpoints(args.run_root))
    in_process = {arm: run_one(arm, args.run_root, args.train_cache) for arm in arms}
    print(f"in-process done: {len(in_process)} arms", flush=True)
    child = subprocess.run(
        [
            sys.executable,
            str(Path(__file__)),
            "--all-fingerprints",
            "--run-root",
            str(args.run_root),
            "--train-cache",
            str(args.train_cache),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=dict(os.environ, PYTHONPATH=str(ROOT), PYTHONUTF8="1"),
        timeout=3600,
    )
    fresh: dict[str, str] = {}
    if child.returncode == 0 and child.stdout.strip():
        fresh = json.loads(child.stdout.strip().splitlines()[-1]).get("digests", {})
    print(f"fresh process done: rc={child.returncode} arms={len(fresh)}", flush=True)
    receipts = [
        {
            "arm": arm,
            "bound_revision": revisions.get(arm),
            "endpoint": str(Path(endpoints(args.run_root)[arm]["path"])).replace("\\", "/"),
            "fingerprint": in_process[arm],
            "fresh_process": {
                "rc": child.returncode,
                "digest": fresh.get(arm),
                "matches_in_process": fresh.get(arm) == in_process[arm]["digest"],
                "stderr_tail": "" if fresh.get(arm) else (child.stderr or "")[-400:],
            },
        }
        for arm in arms
    ]
    for item in receipts:
        print(
            f"{item['arm']:<34} bound={str(item['bound_revision'])[:9]} "
            f"fresh_match={item['fresh_process']['matches_in_process']}",
            flush=True,
        )

    report = {
        "format": "taiji-n177-6-t1-endpoint-receipts-v2",
        "date": "2026-10-29",
        "scope": "15 endpoints, fixed first train episode only; holdout/query never read",
        "instrument_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "revisions": revisions,
        "distinct_revisions": sorted({rev for rev in revisions.values() if rev}),
        "receipts": receipts,
        "all_fresh_match": all(item["fresh_process"]["matches_in_process"] for item in receipts),
        "all_isolated": all(
            item["fingerprint"]["scoring_isolation"]["unchanged"] for item in receipts
        ),
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_report.with_name(args.out_report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, args.out_report)
    print(f"ALL_FRESH_MATCH={report['all_fresh_match']} out={args.out_report.name}")
    print(f"ALL_ISOLATED={report['all_isolated']} instrument={report['instrument_sha256'][:12]}")
    return 0 if report["all_fresh_match"] and report["all_isolated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
