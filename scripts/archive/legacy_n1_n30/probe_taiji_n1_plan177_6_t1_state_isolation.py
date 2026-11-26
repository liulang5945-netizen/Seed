"""N177.6 E0：15 臂写入/更新权限矩阵 + 5 模式有效 mask 的实测（只读，不 step）。

三件读数：
①checkpoint 里自带的 `policy`（及 `policy_sha256`）与当前代码的 `_mode_policy(mode)` 是否一致；
②每模式 `profile_backward_only(batch_index=0)`——它是**前向+反向但不 step** 的现成接缝，
  这里再用"参数字节/优化器状态/cursor 前后相等"复核它确实没写；
③RNG 状态变化单独披露：反向路径可能推进全局 RNG，这不属于"写模型"，但读者要知道。

判据从 policy 的 scope 推出（不是从模式名硬编）：禁入组必须为空、required 组必须有正梯度、
producer 有/无梯度必须与 `producer_updates` 一致。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training.verify_taiji_n1_plan177_6_t1_endpoint_receipts import (  # noqa: E402
    isolation_snapshot,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_t1_course import build_joint_t1_split, joint_t1_split_digest  # noqa: E402
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1  # noqa: E402
from taiji.joint_t1_trainer import (  # noqa: E402
    JOINT_T1_MODES,
    JOINT_T1_TRAINER_COMPONENT_KEY,
    JointT1TrainerV1,
    _mode_policy,
)

RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027"
TRAIN_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
REPORT = ROOT / "reports/taiji_n1_plan177_6_t1_state_isolation_20261029.json"
POLICY_FIELDS = (
    "producer_parameter_owner",
    "producer_updates",
    "language_parameter_owner",
    "language_updates",
    "language_parameter_scope",
    "cognitive_state_writes",
)

_CACHE: JointT1TraceCacheV1 | None = None


def train_cache(path: Path = TRAIN_CACHE) -> JointT1TraceCacheV1:
    global _CACHE
    if _CACHE is None:
        _CACHE = JointT1TraceCacheV1.from_payload(
            torch.load(path, map_location="cpu", weights_only=False)
        )
    return _CACHE


def declared_groups(policy: dict[str, Any], all_groups: set[str]) -> set[str]:
    scope = policy["language_parameter_scope"]
    if scope == "all":
        return set(all_groups)
    if scope == "output_only":
        return {"output"}
    if scope == "none":
        return set()
    raise ValueError(f"unknown language parameter scope: {scope}")


def scope_ok(
    policy: dict[str, Any],
    measured: dict[str, float],
    producer_norm: float,
    all_groups: set[str],
) -> bool:
    """实测算出的梯度分布必须能被 policy 解释；两个方向都要查。"""

    allowed = declared_groups(policy, all_groups)
    if not set(measured) <= allowed:
        return False
    if any(value <= 0.0 for value in measured.values()):
        return False
    if policy["language_parameter_scope"] != "none" and "output" not in measured:
        return False
    return (producer_norm > 0.0) is bool(policy["producer_updates"])


def endpoints(run_root: Path) -> dict[str, dict[str, Any]]:
    state = json.loads((run_root / "run_state.json").read_text(encoding="utf-8"))
    return dict(state["latest_checkpoint_by_arm"])


def _trainer(arm: str, run_root: Path) -> JointT1TrainerV1:
    payload = torch.load(
        Path(endpoints(run_root)[arm]["path"]), map_location="cpu", weights_only=False
    )
    return JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=train_cache())


def policy_matrix(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for arm in sorted(endpoints(run_root)):
        path = Path(endpoints(run_root)[arm]["path"])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        component = payload["components"][JOINT_T1_TRAINER_COMPONENT_KEY]
        policy = dict(component["policy"])
        mode = component["mode"]
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "seed": component["seed"],
                "update_cursor": component["update_cursor"],
                "policy": {key: policy.get(key) for key in POLICY_FIELDS},
                "policy_sha256_matches": content_digest(policy) == component["policy_sha256"],
                "policy_equals_mode_table": policy == _mode_policy(mode),
                "train_split_sha256": component.get("train_split_sha256"),
                "input_trace_cache_sha256": component.get("input_trace_cache_sha256"),
            }
        )
    return rows


def backward_profiles(run_root: Path, batch_index: int) -> list[dict[str, Any]]:
    profiles = []
    for mode in JOINT_T1_MODES:
        arm = f"{mode}__seed_0"
        if arm not in endpoints(run_root):
            continue
        trainer = _trainer(arm, run_root)
        all_groups = {name.split(".", 1)[0] for name, _ in trainer.model.named_parameters()}
        before = isolation_snapshot(trainer)
        rng_before = torch.random.get_rng_state().clone()
        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        result = trainer.profile_backward_only(batch_index=batch_index)
        cpu_seconds = time.process_time() - cpu_before
        wall_seconds = time.perf_counter() - wall_before
        after = isolation_snapshot(trainer)
        rng_after = torch.random.get_rng_state().clone()
        measured = dict(result["language_gradient_norms_by_group"])
        profiles.append(
            {
                "mode": mode,
                "arm": arm,
                "batch_index": batch_index,
                "episode_ids": list(result["episode_ids"]),
                "language_parameter_groups": sorted(all_groups),
                "policy": {
                    "producer_updates": bool(trainer.policy["producer_updates"]),
                    "language_parameter_scope": trainer.policy["language_parameter_scope"],
                    "cognitive_state_writes": bool(trainer.policy["cognitive_state_writes"]),
                },
                "measured": {
                    "producer_gradient_norm": float(result["producer_gradient_norm"]),
                    "language_gradient_norms_by_group": measured,
                },
                "losses": {
                    "parser_loss_nats": float(result["parser_loss_nats"]),
                    "answer_loss_nats_per_byte": float(result["answer_loss_nats_per_byte"]),
                    "answer_target_byte_exposures": int(result["answer_target_byte_exposures"]),
                },
                "mask_matches_declared_scope": scope_ok(
                    trainer.policy,
                    measured,
                    float(result["producer_gradient_norm"]),
                    all_groups,
                ),
                "isolation": {"before": before, "after": after, "unchanged": before == after},
                "rng_state_unchanged": bool(torch.equal(rng_before, rng_after)),
                "cost": {"process_cpu_seconds": cpu_seconds, "wall_seconds": wall_seconds},
            }
        )
        del trainer
    return profiles


def build(run_root: Path, batch_index: int) -> dict[str, Any]:
    rows = policy_matrix(run_root)
    profiles = backward_profiles(run_root, batch_index)
    real_split_sha256 = joint_t1_split_digest(build_joint_t1_split("train"))
    real_cache_sha256 = train_cache().cache_sha256
    verdicts = {
        "fifteen_arms_listed": len(rows) == 15,
        "every_arm_policy_digest_matches": all(row["policy_sha256_matches"] for row in rows),
        "every_arm_policy_equals_mode_table": all(row["policy_equals_mode_table"] for row in rows),
        "every_arm_bound_to_the_real_train_split": all(
            row["train_split_sha256"] == real_split_sha256 for row in rows
        ),
        "every_arm_bound_to_the_real_train_cache": all(
            row["input_trace_cache_sha256"] == real_cache_sha256 for row in rows
        ),
        "five_modes_profiled": len(profiles) == len(JOINT_T1_MODES),
        "every_mode_mask_matches_declared_scope": all(
            item["mask_matches_declared_scope"] for item in profiles
        ),
        "every_mode_left_parameters_optimizer_cursor_unchanged": all(
            item["isolation"]["unchanged"] for item in profiles
        ),
    }
    return {
        "format": "taiji-n177-6-t1-state-isolation-v2",
        "date": "2026-10-29",
        "scope": (
            "read-only: checkpoint policy self-report + backward-only owner profile on "
            "train batch 0; no optimizer step, no retraining, holdout/query never read"
        ),
        "run_root": str(run_root).replace("\\", "/"),
        "train_cache": str(TRAIN_CACHE).replace("\\", "/"),
        "real_train_split_sha256": real_split_sha256,
        "real_train_cache_sha256": real_cache_sha256,
        "batch_index": batch_index,
        "policy_matrix": rows,
        "backward_profiles": profiles,
        "rng_state_unchanged_disclosure": {
            item["mode"]: item["rng_state_unchanged"] for item in profiles
        },
        "verdicts": verdicts,
        "all_verdicts_ok": all(verdicts.values()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177.6 E0 状态隔离/有效 mask 实测（只读）")
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--out-report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = build(args.run_root, args.batch_index)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_report.with_name(args.out_report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.out_report)
    for key, value in report["verdicts"].items():
        print(f"{key}={value}")
    print(f"ALL_VERDICTS_OK={report['all_verdicts_ok']} out={args.out_report.name}")
    return 0 if report["all_verdicts_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
