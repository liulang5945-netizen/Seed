"""A2.5 免训练候选定价（零训练）：把**事件选择**换成"提问与告知共享字符"这把键，表层命中变多少。

前件（两份读数，都不在本件里）：
* `SPEC-A-17` §13——扩展集 48 道带干扰未命中题上，"只摘掉库里那条干扰告知"能把 **17 题**发出来
  （无干扰家族在同一处理下 0/33＝恒等对照）⇒ 事件选择的价格已量清；
* `reports/taiji_r2_a25_cue_separability_extension_20260926.json`——同一批题上**静态**挑事件，
  现状神经 cue 挑对 **37/48**，字符重叠键挑对 **48/48**。

所以"要不要给选择加可学参数"之前先问：**光换查询能拿回多少表层命中**。本件就是那一刀。
对照不用重跑——同一支记分件在**未打补丁**时的冻结读数（`taiji_r2_copy_surface_extension_20260925.json`：
seed-A 23/104、seed-B 22/104、对照 0/104），本件与它同链同底同题集（基座/电路/题集摘要三者都比过）。

纪律：**不改产品代码**。只在进程内把 `ToldContentStore.best_match`（生产发射实际消费的那一个接口，
`taiji/copy_circuit.py:259`）换成规则版，并且按**被 import 的那个名字**包 `_answer_raw` 来传入当前提问
——`run_arm` 的题循环、判命中口径与冻结读数一字不动（补丁钉在消费的接口上，不钉实现行）。
零训练（`learn=False` 由产品链决定）、`checkpoints/` 只读且跑前后 sha256 复核、只写 `reports/` 一份新件。

**两种对照**：跑 v1 用外部冻结读数做对照（并先未打补丁重跑一臂、逐题核对它真的复现了件里的数）；
跑 **v2（难干扰集）没有冻结读数**，于是 `--paired`：每臂先未打补丁现跑一遍当对照，再跑规则键。

**这把键不是产品机制，读数是"规则基线"**：它按字面共享字符排序，对间接指代天然无效
（§9 在同一把键上测出 0.583——"我叫阿岩。"→"我的名字是什么？"）。所以本件的值是
**任何学习式选择器必须先超过的那条线**，不是"选问题已经解决"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
#: 冻结的未打补丁读数（同链同底同题集）——本件拿它做对照，不重跑。
BASELINE_REPORT = PROJECT_ROOT / "reports/taiji_r2_copy_surface_extension_20260925.json"
SEED_A_CIRCUIT = "output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
SEED_B_CIRCUIT = "output/taiji_r2_copy_circuit_chat_seedB/judge/circuit-final.pt"

#: 补丁现场计数（每臂跑完即取并清零，避免把上一臂的调用算进这一臂）。
#: `no_query`＝**库里有得挑却没拿到提问**（真缺陷）；`empty_store`＝库是空的，
#: 原实现同样返回 `None`，属良性退回，不算覆盖缺口。混成一个数会把"没有可挑"读成"没打上补丁"。
PROBE: dict[str, int | None] = {"query": None, "used": 0, "no_query": 0, "empty_store": 0}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _overlap_best_match(self: Any, cue: Any) -> Any:
    """规则版事件选择：与"当前提问"共享字符最多的一条告知胜出。

    平手按 `event_id` 取新的一条——与 §9 那支可分离探针里 `oracle` 档的排序规则**一致**，
    两处不一致的话，静态 48/48 与表层增益就不是同一把键的读数。
    """
    original = _overlap_best_match.original  # type: ignore[attr-defined]
    query = PROBE["query"]
    events = self.events()
    if not events:
        PROBE["empty_store"] = int(PROBE["empty_store"] or 0) + 1
        return original(self, cue)
    if not query:
        #: 有两条以上告知却没拿到当前提问——这一格走的是**原键**，整臂读数就混进了
        #: 未打补丁的行为，必须让覆盖检查响亮失败（退回不静默）。
        PROBE["no_query"] = int(PROBE["no_query"] or 0) + 1
        return original(self, cue)
    scored = sorted(
        (
            (
                len(set(str(query)) & set(event.content.decode("utf-8", errors="replace"))),
                int(event.event_id),
                event,
            )
            for event in events
        ),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    PROBE["used"] = int(PROBE["used"] or 0) + 1
    return scored[0][2]


def _first_event_selection(self: Any, cue: Any) -> Any:
    """第三档＝**天花板**：两条告知都留在库里，但选择固定取最早入库的那条。

    v1/v2 的题形都是「告知 → 干扰 → 提问」，`record_told_history` 按历史顺序入库，
    所以"最早那条"就是含答案的告知——这一档与 §13 的 `--store target` 消融同义
    （那条是把干扰**删掉**，这条是留着但永远不选它），差别是这里不碰库、只碰选择。
    """
    events = self.events()
    if not events:
        return None
    return min(events, key=lambda event: int(event.event_id))


def _answer_raw_with_query(runtime: Any, prompt: str, history: Any) -> str:
    """把"当前提问"交给规则键，其余原样转给产品那支 `_answer_raw`。"""
    PROBE["query"] = prompt
    return _answer_raw_with_query.original(runtime, prompt, history)  # type: ignore[attr-defined]


def _install(selector: str = "overlap") -> None:
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

    from taiji.copy_circuit import ToldContentStore

    _answer_raw_with_query.original = _answer_raw
    sys.modules["score_taiji_r2_copy_circuit_chat_cap"]._answer_raw = _answer_raw_with_query
    _overlap_best_match.original = ToldContentStore.best_match
    ToldContentStore.best_match = (
        _overlap_best_match if selector == "overlap" else _first_event_selection
    )


def _uninstall() -> None:
    import score_taiji_r2_copy_circuit_chat_cap as cap_module

    from taiji.copy_circuit import ToldContentStore

    cap_module._answer_raw = _answer_raw_with_query.original  # type: ignore[attr-defined]
    ToldContentStore.best_match = _overlap_best_match.original  # type: ignore[attr-defined]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--arms",
        default=",".join([SEED_A_CIRCUIT, SEED_B_CIRCUIT]),
        help="逗号分隔的电路路径（默认两个独立初始化的电路，与冻结读数同一对）",
    )
    parser.add_argument("--manifest", default=str(MANIFEST), help="题集清单（默认 v1 冻结件）")
    parser.add_argument(
        "--baseline-report",
        default=str(BASELINE_REPORT),
        help="未打补丁的冻结读数件；传 `none` 表示没有外部对照（新题集），此时必须 `--paired`",
    )
    parser.add_argument(
        "--selector",
        choices=("overlap", "first"),
        default="overlap",
        help="overlap＝与提问共享字符的规则键；first＝天花板（永远选最早入库那条＝§13 的摘干扰消融）",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="每臂都先未打补丁跑一遍当对照（没有冻结读数时唯一能给出差值的办法）",
    )
    parser.add_argument(
        "--verify-arm",
        default=SEED_A_CIRCUIT,
        help="有冻结读数时先把这一臂未打补丁重跑与件里对账；空串跳过（跳过＝对照未核验，读数降级）",
    )
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from score_taiji_r2_copy_surface_extension import load_items, run_arm

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    manifest_sha = _sha256(manifest_path)

    baseline: dict[str, Any] | None = None
    verify_target = ""
    if args.baseline_report != "none":
        baseline_path = PROJECT_ROOT / args.baseline_report
        if not baseline_path.is_absolute():
            baseline_path = PROJECT_ROOT / args.baseline_report
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if manifest_sha != str(baseline.get("manifest_sha256")):
            #: 题集换过，对照读数就不可比——先响，别顺跑出第三份数。
            print(
                json.dumps(
                    {
                        "error": "题集摘要与冻结读数不符",
                        "manifest": manifest_sha[:12],
                        "baseline": str(baseline.get("manifest_sha256"))[:12],
                    }
                )
            )
            return 1
        if str(baseline.get("checkpoint")) != args.checkpoint:
            print(json.dumps({"error": "基座与冻结读数不是同一个底，不可比"}))
            return 1
        verify_target = args.verify_arm
    elif not args.paired:
        print(json.dumps({"error": "没有外部对照时必须 --paired，否则差值没有对照物"}))
        return 1

    items = load_items(manifest_path)
    #: 冻结读数里没有电路摘要（`treated_arms[]` 只有 `circuit` 路径），所以"同底同链"不能靠摘要比——
    #: 改成**先把未打补丁的那臂重跑一遍**，要求严格命中与件里逐题相同。不相等就说明现场变了，
    #: 这时"补丁带来的差"没有对照物，直接退出而不是顺跑出第三份数。
    unpatched: dict[str, Any] = {"performed": False}
    if verify_target and baseline is not None:
        verify_arm = run_arm(items, checkpoint, verify_target)
        base_match = next(
            (arm for arm in baseline["treated_arms"] if str(arm.get("circuit")) == verify_target),
            None,
        )
        if base_match is None:
            print(json.dumps({"error": f"冻结读数里没有臂 {verify_target}"}))
            return 1
        expected = int(base_match["strict_hits"])
        unpatched = {
            "performed": True,
            "circuit": verify_target,
            "strict_hits": verify_arm["strict_hits"],
            "baseline_strict_hits": expected,
            "reproduces": bool(verify_arm["strict_hits"] == expected),
            "rows_match": [(str(row["id"]), bool(row["hit"])) for row in verify_arm["rows"]]
            == [(str(row["id"]), bool(row["hit"])) for row in base_match["rows"]],
        }
        if not unpatched["reproduces"]:
            print(
                json.dumps({"error": "未打补丁的重跑与冻结读数不符，对照失效", "check": unpatched})
            )
            return 3

    def _baseline_for(circuit: str) -> dict[str, int]:
        """取这一臂的未打补丁读数：优先用冻结件，没有就**现跑一遍**（`--paired`）。"""
        if baseline is not None:
            found = next(
                (arm for arm in baseline["treated_arms"] if str(arm.get("circuit")) == circuit),
                None,
            )
            if found is None:
                raise KeyError(f"冻结读数里没有这一臂，无从比差：{circuit}")
            return {
                "strict_hits": int(found["strict_hits"]),
                "well_formed_texts": int(found["well_formed_texts"]),
            }
        control_arm = run_arm(items, checkpoint, circuit)
        return {
            "strict_hits": int(control_arm["strict_hits"]),
            "well_formed_texts": int(control_arm["well_formed_texts"]),
        }

    arms: list[dict[str, Any]] = []
    for circuit in [part.strip() for part in args.arms.split(",") if part.strip()]:
        if not (PROJECT_ROOT / circuit).exists():
            print(json.dumps({"error": f"电路不存在：{circuit}"}))
            return 1
        base_arm = _baseline_for(circuit)
        PROBE["used"] = 0
        PROBE["no_query"] = 0
        PROBE["empty_store"] = 0
        _install(args.selector)
        try:
            arm = run_arm(items, checkpoint, circuit)
        finally:
            _uninstall()
        used = int(PROBE["used"] or 0)
        no_query = int(PROBE["no_query"] or 0)
        empty_store = int(PROBE["empty_store"] or 0)
        arms.append(
            {
                "circuit": circuit,
                "circuit_sha256": _sha256(PROJECT_ROOT / circuit),
                "selector_used_calls": used,
                "selector_no_query_calls": no_query,
                "selector_empty_store_calls": empty_store,
                "items": arm["items"],
                "texts": arm["texts"],
                "strict_hits": arm["strict_hits"],
                "baseline_strict_hits": base_arm["strict_hits"],
                "strict_hits_delta": arm["strict_hits"] - int(base_arm["strict_hits"]),
                "well_formed_texts": arm["well_formed_texts"],
                "baseline_well_formed_texts": base_arm["well_formed_texts"],
                "utf8_decodable_rate": arm["utf8_decodable_rate"],
                "rows": arm["rows"],
            }
        )

    #: 库里可挑却没拿到提问 ⇒ 那一格走的还是原键，整臂读数就不是"规则键的数"。
    #: （空库的良性退回另记，见 `_overlap_best_match`。）
    selector_coverage_complete = all(
        int(arm["selector_no_query_calls"]) == 0 and int(arm["selector_used_calls"]) > 0
        for arm in arms
    )
    report = {
        "format": "taiji-r2-a25-overlap-selector-price-v2",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §13/§15",
        "patched": [
            "taiji.copy_circuit.ToldContentStore.best_match（消费点 taiji/copy_circuit.py:259）",
            "score_taiji_r2_copy_circuit_chat_cap._answer_raw（按被 import 的名字包，用来传当前提问）",
        ],
        "checkpoint": args.checkpoint,
        "manifest": (
            manifest_path.relative_to(PROJECT_ROOT).as_posix()
            if manifest_path.is_relative_to(PROJECT_ROOT)
            else str(manifest_path)
        ),
        "manifest_sha256": manifest_sha,
        "selector": args.selector,
        "paired": bool(baseline is None),
        "baseline_report": (
            "none" if baseline is None else str(args.baseline_report).replace("\\", "/")
        ),
        "unpatched_reproduction": unpatched,
        "selector_coverage_complete": selector_coverage_complete,
        "arms": arms,
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_r2_a25_overlap_selector_price_20260926.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arms": [
                    {
                        "circuit": arm["circuit"],
                        "hits": arm["strict_hits"],
                        "baseline": arm["baseline_strict_hits"],
                        "delta": arm["strict_hits_delta"],
                        "used": arm["selector_used_calls"],
                        "no_query": arm["selector_no_query_calls"],
                        "empty_store": arm["selector_empty_store_calls"],
                    }
                    for arm in arms
                ],
                "manifest": report["manifest"],
                "selector": args.selector,
                "paired": report["paired"],
                "unpatched_reproduction": unpatched,
                "coverage_complete": selector_coverage_complete,
                "base_unchanged": report["base_sha256_unchanged"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    if not selector_coverage_complete or not report["base_sha256_unchanged"]:
        return 2
    #: 有外部冻结对照时必须真做过对账；`--paired` 时对照就是本次现跑的未打补丁臂，天然成立。
    if baseline is not None and not bool(unpatched.get("performed")) and args.verify_arm:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
