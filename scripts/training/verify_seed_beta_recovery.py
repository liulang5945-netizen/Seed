#!/usr/bin/env python3
"""公测（M3）门槛：检查点崩溃恢复可靠性演练。

判据（公测路线图 §1.3）：
    模拟崩溃恢复 10/10 成功；原子落盘（临时文件 + rename），无半写文件。

每轮演练三个故障注入场景：
1. 序列化中途崩溃（``torch.save`` 写到一半抛错）——目标文件必须保持
   上一个完好版本，且目录无 ``.tmp`` 残留。
2. 半写临时文件遗留（模拟进程被杀）——下一次成功保存必须干净替换，
   恢复后状态与崩溃前一致。
3. 目标文件被截断破坏（模拟磁盘事故）——``restore`` 必须明确报错而不是
   静默加载坏状态；随后一次成功保存恢复可用性。

每轮判过 = 三场景全部通过 + 恢复后的 ``score_bytes`` 与崩溃前逐位一致。

注：``torch.save`` 的 zip 存档字节每次保存都会变（archive 名含对象 id），
因此“干净替换”判定走加载后状态等价，而不是原始字节相等。
输出 ``reports/seed_beta_recovery_<date>.json``，失败以非零码退出。

运行：python -X utf8 -u scripts/training/verify_seed_beta_recovery.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import _verify_emit  # noqa: E402
import torch  # noqa: E402

from seed import Seed, SeedConfig  # noqa: E402
from seed.persistence import atomic_save, attach_metadata  # noqa: E402

ROUNDS = 10
PROBE = "问：你好。\n答：你好，很高兴见到你。".encode()


class _MidSerializationCrash:
    """pickle 序列化到一半抛错的载荷（模拟写入中途崩溃）。

    ``__reduce_ex__`` 在 dump 阶段被调用，此时临时文件已经写入若干字节，
    抛错即复现“写了一半进程死掉”的现场。
    """

    def __reduce_ex__(self, protocol):
        raise RuntimeError("simulated crash mid-serialization")


def _fresh_model(round_index: int) -> Seed:
    model = Seed(SeedConfig(), episode_id=f"recovery-{round_index}")
    model.learn_bytes(PROBE + str(round_index).encode("ascii"))
    return model


def _tmp_residue(target: Path) -> list[str]:
    return [p.name for p in target.parent.iterdir() if p.name.endswith(".tmp")]


def _state_matches(target: Path, model: Seed, score_before) -> bool:
    """加载目标检查点，判定状态与基准模型语义等价。"""
    restored = Seed.from_checkpoint(torch.load(target, map_location="cpu", weights_only=False))
    return restored.tick == model.tick and restored.score_bytes(PROBE) == score_before


def _run_round(round_index: int, target: Path) -> dict[str, object]:
    results: dict[str, object] = {}

    # 场景 0：正常落盘，记录崩溃前状态。
    model = _fresh_model(round_index)
    envelope = attach_metadata(
        model.checkpoint(), tick=model.tick, extra={"trainer": "recovery-drill"}
    )
    atomic_save(envelope, target)
    score_before = model.score_bytes(PROBE)
    good_bytes = target.read_bytes()

    # 场景 1：序列化中途崩溃。
    try:
        atomic_save({"envelope": envelope, "broken": _MidSerializationCrash()}, target)
        crashed = False
    except RuntimeError:
        crashed = True
    results["crash_mid_save_raises"] = crashed
    results["target_survives_mid_save_crash"] = target.read_bytes() == good_bytes
    results["no_tmp_residue_after_crash"] = _tmp_residue(target) == []

    # 场景 2：半写临时文件遗留 + 下一次保存干净替换。
    stray = target.parent / (target.name + ".stray.tmp")
    stray.write_bytes(b"\x00" * 4096)
    results["restore_after_crash_matches"] = _state_matches(target, model, score_before)
    atomic_save(envelope, target)
    results["clean_replace_after_stray_tmp"] = (
        _state_matches(target, model, score_before) and not stray.exists()
    )

    # 场景 3：目标文件截断破坏。
    corrupted = good_bytes[: len(good_bytes) // 2]
    target.write_bytes(corrupted)
    try:
        Seed.from_checkpoint(torch.load(target, map_location="cpu", weights_only=False))
        rejected = False
    except Exception:
        rejected = True
    results["corrupted_target_rejected"] = rejected
    atomic_save(envelope, target)
    results["service_restored_after_repair"] = _state_matches(target, model, score_before)

    round_pass = all(
        bool(value) for key, value in results.items() if key != "crash_mid_save_raises"
    ) and bool(results["crash_mid_save_raises"])
    results["round_pass"] = round_pass
    # 清场，下一轮用全新目标。
    target.unlink(missing_ok=True)
    for stray in target.parent.glob(target.name + "*.tmp"):
        stray.unlink(missing_ok=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workdir = REPO / "data" / "_tmp_recovery_drill"
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "drill_checkpoint.pt"

    rounds = {}
    for index in range(args.rounds):
        rounds[f"round_{index + 1}"] = _run_round(index, target)
        print(
            f"[recovery] round {index + 1}: "
            f"{'PASS' if rounds[f'round_{index + 1}']['round_pass'] else 'FAIL'}",
            flush=True,
        )
    workdir.rmdir()

    passed = sum(1 for row in rounds.values() if row["round_pass"])
    checks = {
        "crash_recovery_10_of_10": passed == args.rounds,
        "no_tmp_residue_any_round": all(
            row["no_tmp_residue_after_crash"] for row in rounds.values()
        ),
    }
    report = {
        "benchmark": "seed_beta_recovery",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rounds": rounds,
        "metrics": {"passed_rounds": passed, "total_rounds": args.rounds},
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    out_path = args.output or (
        REPO / "reports" / f"seed_beta_recovery_{time.strftime('%Y%m%d')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered + "\n", encoding="utf-8")
    print(f"\n报告已写入 {out_path}", file=sys.stderr)
    return _verify_emit.emit_and_exit("seed_beta_recovery", report)


if __name__ == "__main__":
    raise SystemExit(main())
