"""PLAN-N2-03 的 Arm R（收束臂）驱动：**只装载→只收束→只存盘**，全程不 observe、不 learn、不调 sleep_pass。

为什么要这一臂：run-2 的代价（7 列跌破）有两族可能的来源——巩固的权重更新，与 DEBT-G47 修法甲那一步
`reset_dynamics`（它按代码事实**整块换掉 `_state`**，情节/工作记忆 7,148 个叶子因此在候选档里消失）。
这一臂把后者单独拿出来跑同一批面，归因分数由判读器算（本驱动只产档与自述，不产结论）。

三条硬事实必须由本驱动自己证（件 §3 G-N6-1/2）：
* `reset_dynamics` **不改权重摘要**（前后＋母档三者逐位同）；
* 存出来的臂档**可被产品自己的装载器读回**，且读回摘要仍等于母档摘要；
* 除"收束"之外什么都没做：驱动不 import `sleep_pass`，也不给任何 `learn=True` 的入口。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))

from run_taiji_n2_powerup import _digest  # noqa: E402 - 摘要口径与两次通电同一式子

DEFAULT_SOURCE = PROJECT_ROOT / "checkpoints" / "seed_n2b_mother_20261008.pt"
DEFAULT_OUT = PROJECT_ROOT / "output" / "n2b_resetarm" / "checkpoint.pt"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_n2b_resetarm_20261008.json"


def _wake_episode_id() -> str:
    """与 `seed_platform.sleep_pass.WAKE_EPISODE_ID` 同一个常量、同一式子。

    不另起一个名字——否则"收束到哪一态"就不是同一件事了（跨臂同式子才可归因）。
    """

    from seed_platform.sleep_pass import WAKE_EPISODE_ID

    return WAKE_EPISODE_ID


#: **参数族 vs 情节/活动族**的分界（PLAN-N2-03 §3ter 的口径）：初稿拿 `content_digest(checkpoint())`
#: 当"权重摘要"，第一次实跑就红（3e00bafc… → 7e3e640a…），红因是定义错而非产品意外——
#: 那个摘要覆盖整份信封，而 `reset_dynamics` 操作的正是信封里的情节/活动族。
#: ⇒ 守卫改成只对**参数族**要求零差；情节/活动族的变化是本臂的处理本身，必须报出、不得拒判。
STATE_PREFIXES = (
    "substrate.cognitive_state",
    "taiji.cognitive_state",
    "substrate.state",
    "taiji.kernel.state",
    "substrate.perception",
    "taiji.components.perception",
)

#: 第二处口径排除（**连值一起报出，不静默放宽**）：实测参数族里只有 2 条差异，且是同一件东西的两份信封副本——
#: `…identity_organ.lineage.parent_checkpoint_digest`（＝**上一份档的 sha256**，随序列化自动更新，不是学出来的参数）。
#: 本机实测值：`61a431eb304ebb18…` → `1be216de20917c70…`（两条路径同值）。
#: 这类"信封自指的血缘记账"正是 DEBT-G47 那一族的成员，用它当"权重变没变"的判据会永远为假；
#: 但它必须出现在件里（`excluded_changed_paths`），否则就成了为了让守卫绿而改守卫。
LINEAGE_SUFFIXES = (".lineage.parent_checkpoint_digest", ".lineage.checkpoint_digest")


def _is_parameter_path(path: str) -> bool:
    return not path.startswith(STATE_PREFIXES) and not path.endswith(LINEAGE_SUFFIXES)


def _family_leaves(model: Any) -> dict[str, Any]:
    """把 `model.checkpoint()` 摊平成 `{路径: 叶子}`，按参数族／情节活动族分开。"""

    out: dict[str, Any] = {}

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            if not node:
                out[path + "{}"] = None
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            if not node:
                out[path + "[]"] = None
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif torch.is_tensor(node):
            out[path] = node
        else:
            out[path] = node

    walk(model.checkpoint(), "")
    params = {k: v for k, v in out.items() if _is_parameter_path(k)}
    state = {k: v for k, v in out.items() if not _is_parameter_path(k)}
    return {"params": params, "state": state}


def _lineage_bookkeeping(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """把"被排除但确实变了"的信封记账条目原样报出来（含前后值）——守卫的每一次收窄都要留痕。"""

    out: dict[str, Any] = {}
    for key in sorted(set(before["state"]) & set(after["state"])):
        if not key.endswith(LINEAGE_SUFFIXES):
            continue
        x, y = before["state"][key], after["state"][key]
        if x != y:
            out[key] = {"before": str(x)[:70], "after": str(y)[:70]}
    return out


def _params_digest(families: dict[str, Any]) -> str:
    from taiji.internalization import content_digest

    plain = {
        key: (value.detach().clone() if torch.is_tensor(value) else value)
        for key, value in families["params"].items()
    }
    return content_digest(plain)


def _family_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """参数族要求零差；情节/活动族只**报数**（这正是这一臂做的事）。"""

    def compare(side: str) -> dict[str, Any]:
        a, b = before[side], after[side]
        common = sorted(set(a) & set(b))
        changed = 0
        for key in common:
            x, y = a[key], b[key]
            if torch.is_tensor(x):
                if x.shape != y.shape or x.numel() and not torch.equal(x, y):  # type: ignore[union-attr]
                    changed += 1
            elif x != y:
                changed += 1
        return {
            "entries_before": len(a),
            "entries_after": len(b),
            "changed": changed,
            "dropped": len(set(a) - set(b)),
            "added": len(set(b) - set(a)),
        }

    return {"parameter_family": compare("params"), "state_family": compare("state")}


def run_arm(source: Path, out: Path, report: Path) -> int:
    started = time.time()
    from api.seed_runtime import SeedRuntime

    payload: dict[str, Any] = {
        "format": "taiji-n2b-resetarm-v1",
        "prereg": "plans/reference/PLAN-N2-03_reset_attribution_prereg_20261008.md#1",
        "source": str(source),
        "arm_path": str(out),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
    }
    runtime = SeedRuntime.load(source)
    mother_digest = _digest(runtime.model)
    before = _family_leaves(runtime.model)
    mother_params = _params_digest(before)
    mother_tick = int(runtime.model.tick)
    episode = _wake_episode_id()

    #: 这一臂唯一的动作——与 `sleep_pass` 收束那一步同一式子、同一常量。
    runtime.model.reset_dynamics(episode_id=episode)
    after = _family_leaves(runtime.model)
    arm_params = _params_digest(after)
    diff = _family_diff(before, after)
    payload.update(
        {
            "wake_episode_id": episode,
            "mother_digest": mother_digest,
            "digest_after_reset": _digest(runtime.model),
            "params_digest_before": mother_params,
            "params_digest_after": arm_params,
            "g_n6_1_parameter_family_unchanged": mother_params == arm_params,
            "family_diff": diff,
            "excluded_changed_paths": _lineage_bookkeeping(before, after),
            "exclusion_rule": {
                "state_prefixes": list(STATE_PREFIXES),
                "lineage_suffixes": list(LINEAGE_SUFFIXES),
                "why": "参数族只含学出来的参数；情节/活动族与信封血缘记账按 §3ter 单独报出，不混进守卫",
            },
            "mother_tick": mother_tick,
            "tick_after_reset": int(runtime.model.tick),
        }
    )
    if not payload["g_n6_1_parameter_family_unchanged"]:
        raise RuntimeError(
            f"reset_dynamics 改动了参数族摘要（{mother_params[:16]}… → {arm_params[:16]}…）"
            "⇒ 这一臂不是'只收束'，整件不判，回 owner"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    runtime.save(out)
    #: G-N6-2（同 §3ter 口径）：臂档必须可被**产品自己的装载器**读回，且读回后的**参数族**摘要
    #: 仍等于收束后的参数族摘要。整份信封摘要**不该**相等——情节族被这一臂清空正是处理本身。
    reloaded = SeedRuntime.load(out)
    reloaded_params = _params_digest(_family_leaves(reloaded.model))
    payload.update(
        {
            "arm_params_digest_after_load": reloaded_params,
            "g_n6_2_arm_loadable_and_params_identical": reloaded_params == arm_params,
            "arm_saved_bytes": out.stat().st_size,
            "duration_seconds": round(time.time() - started, 1),
        }
    )
    payload["fail_closed_pass"] = bool(
        payload["g_n6_1_parameter_family_unchanged"]
        and payload["g_n6_2_arm_loadable_and_params_identical"]
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"parameter_family_unchanged={payload['g_n6_1_parameter_family_unchanged']}")
    print(f"arm_loadable_params_identical={payload['g_n6_2_arm_loadable_and_params_identical']}")
    print(f"family_diff={json.dumps(payload['family_diff'])}")
    print(f"arm_bytes={payload['arm_saved_bytes']}")
    print(f"report -> {report}")
    return 0 if payload["fail_closed_pass"] else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2-03 Arm R：只收束、不学习（零改权重）")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    return run_arm(args.source, args.out, args.report)


if __name__ == "__main__":
    raise SystemExit(main())
