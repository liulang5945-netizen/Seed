"""A2 复制回路**回归门**（立项 §4.3）：装上没有让既有能力变差？（零训练，纯比对）

输入两份 `eval_taiji_cap0_baseline.py` 报告：
* ``--control``  默认链路（不挂电路）
* ``--treated``  治疗链路（``--copy-circuit`` 挂载训练后电路）

**先否证可比性，再谈分数**（同链同底；这一步不能省，历史上换链路会让同一判据给出相反结论）：
两报告的 checkpoint 摘要、git HEAD、评价集摘要、除 `copy_circuit` 外的链路字段必须逐项相等，
任一项不等 ⇒ 拒绝判决（`comparable=false`），只如实并列。

判据（冻结自 `M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md` §4.3）：
1. **A/B/C 维不劣化**：`machine_normalised(treated) >= machine_normalised(control)`；
2. **表层字节分布不劣于基线**：UTF-8 可解码率与成句率（n 元模型口径）不下降；
3. **`learn_bytes` 既有合同测试全绿**：由 pytest 面出，本件只声明不代判。
D/E/G 与"逐题翻转清单"作为**观察项**并列（D/E 是这条回路要抬的，不参与"不劣化"判据）。
人工复核项一律记 `unverified`，不许被代理信号点亮成"通过"。
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

#: 有机器分数的"不劣化"判定维（立项 §4.3 列 A/B/C；A 是布尔检查，另走 `--*-health`）。
JUDGED_DIMENSIONS = ("B", "C")
#: 只观察、不判定的维度（D/E 是增益目标，G 大量项靠人工复核）。
OBSERVED_DIMENSIONS = ("D", "E", "G")


def _load(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def _chain(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("chain") or {})


def comparability(
    control: dict[str, Any],
    treated: dict[str, Any],
    *,
    require_eval_set: bool = True,
) -> dict[str, Any]:
    """逐项核对两报告是否同底同链；返回 defects 清单（空＝可比）。"""
    defects: list[str] = []
    ident_c, ident_t = control.get("identity") or {}, treated.get("identity") or {}
    identity_keys = ["git_head", "checkpoint_sha256"]
    if require_eval_set:
        identity_keys.append("eval_set_sha256")
    for key in identity_keys:
        left, right = ident_c.get(key), ident_t.get(key)
        if left is None or right is None:
            defects.append(f"identity.{key} 缺失（有一方没做身份绑定，不能并读）")
        elif left != right:
            defects.append(f"identity.{key} 不一致：{left} != {right}")
    chain_c, chain_t = _chain(control), _chain(treated)
    if chain_c.get("relax_legacy_guard") != chain_t.get("relax_legacy_guard") or chain_c.get(
        "constrained_decode"
    ) != chain_t.get("constrained_decode"):
        defects.append(f"既有链路开关不一致：{chain_c} != {chain_t}")
    if chain_c.get("copy_circuit"):
        defects.append("对照臂自己挂了电路，不是「默认链路」对照")
    if not chain_t.get("copy_circuit"):
        defects.append("治疗臂没挂电路，两臂其实是同一条链")
    if control.get("trained_during_eval") or treated.get("trained_during_eval"):
        defects.append("有臂在评测期间训练了，分数身份不成立")
    return {
        "comparable": not defects,
        "defects": defects,
        "chain_control": chain_c,
        "chain_treated": chain_t,
    }


def _tally(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    block = (report.get("dimensions") or {}).get(key)
    if not isinstance(block, dict):
        return None
    return block.get("tally") if isinstance(block.get("tally"), dict) else block


def _turn_texts(report: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for block in (report.get("dimensions") or {}).values():
        for item in block.get("items", []) if isinstance(block, dict) else []:
            for turn in item.get("turns", []) or []:
                answer = turn.get("raw_output")
                if isinstance(answer, str):
                    texts.append(answer)
    return texts


#: n 元模型只建一次（每条臂各调一次 surface_readout；重建要读语料，白等）。
_NGRAM_MODEL: Any = None


def _ngram_model() -> Any:
    global _NGRAM_MODEL
    if _NGRAM_MODEL is None:
        from eval_taiji_r2_readout_retrain import build_ngram_model

        _NGRAM_MODEL = build_ngram_model()
    return _NGRAM_MODEL


def surface_readout(texts: list[str]) -> dict[str, Any]:
    """表层分布三条：UTF-8 可解码率／含替换符率／成句率（n 元口径，与 R2 仪器同源）。"""
    from eval_taiji_r2_readout_retrain import well_formed

    ngram = _ngram_model()
    total = len(texts)
    if not total:
        return {"texts": 0}
    decodable = sum(
        1
        for text in texts
        if "\ufffd" not in text and not any("\ud800" <= ch <= "\udfff" for ch in text)
    )
    replacement = sum(1 for text in texts if "\ufffd" in text)
    formed = sum(1 for text in texts if well_formed(text, ngram))
    cjk = sum(
        1
        for text in texts
        for ch in text
        if unicodedata.category(ch).startswith("Lo") and "一" <= ch <= "鿿"
    )
    chars = sum(len(text) for text in texts)
    return {
        "texts": total,
        "utf8_decodable_rate": round(decodable / total, 4),
        "replacement_char_rate": round(replacement / total, 4),
        "well_formed_rate": round(formed / total, 4),
        "cjk_share": round(cjk / max(chars, 1), 4),
    }


def _item_status(report: dict[str, Any], key: str) -> dict[str, Any]:
    block = (report.get("dimensions") or {}).get(key) or {}
    out: dict[str, Any] = {}
    for item in block.get("items", []) or []:
        item_id = str(item.get("id"))
        if "pass" in item:
            out[item_id] = bool(item["pass"])
        elif "status" in item:
            out[item_id] = str(item["status"])
    return out


def _a_dimension_check(control: dict[str, Any], treated: dict[str, Any]) -> dict[str, Any]:
    """A 维（模型真实性）没有分数，只有布尔检查项 ⇒ 逐键判"对照里为真的，治疗里也得为真"。

    不这么处理就会踩两类坑：把 A 当成"无 tally ⇒ unverified"放掉，
    或者拿别的东西（比如 H 的计时）当代理信号点亮它。H **不判**：并发负载下的计时不能标定。
    """

    def checks(report: dict[str, Any]) -> dict[str, Any]:
        block = (report.get("dimensions") or {}).get("A") or {}
        return dict(block.get("checks") or {})

    left, right = checks(control), checks(treated)
    if not left or not right:
        return {"status": "not_executed", "counts_as_pass": False}
    lost = sorted(k for k, v in left.items() if v is True and right.get(k) is not True)
    gained = sorted(k for k, v in right.items() if v is True and left.get(k) is not True)
    return {
        "status": "ok" if not lost else "regressed",
        "control_checks": len(left),
        "lost": lost,
        "gained": gained,
        "counts_as_pass": not lost,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, help="默认链路 CAP 报告")
    parser.add_argument("--treated", required=True, help="挂载训练后电路的 CAP 报告")
    parser.add_argument("--circuit-payload", default=None)
    parser.add_argument("--control-health", default=None, help="A 维（真实性检查）对照那份")
    parser.add_argument("--treated-health", default=None, help="A 维治疗那份（同链路＋挂电路）")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    control, treated = _load(Path(args.control)), _load(Path(args.treated))
    gate = comparability(control, treated)
    verdicts: dict[str, Any] = {}
    non_degraded = True
    if args.control_health and args.treated_health:
        health_c = _load(Path(args.control_health))
        health_t = _load(Path(args.treated_health))
        gate["health"] = comparability(health_c, health_t, require_eval_set=False)
        verdicts["A"] = _a_dimension_check(health_c, health_t)
    else:
        verdicts["A"] = {
            "status": "not_executed",
            "counts_as_pass": False,
            "note": "未给 --control-health/--treated-health ⇒ A 维没测，回归门不完整（不记通过）",
        }
    for key in JUDGED_DIMENSIONS:
        left, right = _tally(control, key), _tally(treated, key)
        if left is None or right is None:
            verdicts[key] = {"status": "not_executed", "counts_as_pass": False}
            non_degraded = False
            continue
        left_score = left.get("machine_normalised", left.get("normalised"))
        right_score = right.get("machine_normalised", right.get("normalised"))
        if left_score is None or right_score is None:
            # 人工复核维/未机检维：记 unverified，绝不用代理信号点亮成通过。
            verdicts[key] = {
                "status": "unverified",
                "control": left,
                "treated": right,
                "counts_as_pass": False,
            }
            continue
        ok = float(right_score) >= float(left_score)
        non_degraded = non_degraded and ok
        status_c = _item_status(control, key)
        status_t = _item_status(treated, key)
        verdicts[key] = {
            "status": "ok" if ok else "regressed",
            "control": float(left_score),
            "treated": float(right_score),
            "counts_as_pass": ok,
            #: 逐题翻转清单——总分相同但翻了两题（一涨一跌）时，只有清单能看出来。
            "status_flips": {
                item_id: [status_c.get(item_id), status_t.get(item_id)]
                for item_id in sorted(set(status_c) | set(status_t))
                if status_c.get(item_id) != status_t.get(item_id)
            },
        }
    non_degraded = non_degraded and bool(verdicts["A"].get("counts_as_pass"))
    surface_control = surface_readout(_turn_texts(control))
    surface_treated = surface_readout(_turn_texts(treated))
    surface_ok = bool(
        surface_control.get("texts")
        and surface_treated.get("texts")
        and surface_treated["utf8_decodable_rate"] >= surface_control["utf8_decodable_rate"]
        and surface_treated["well_formed_rate"] >= surface_control["well_formed_rate"]
    )
    observed = {
        key: {"control": _tally(control, key), "treated": _tally(treated, key)}
        for key in OBSERVED_DIMENSIONS
    }
    report = {
        "format": "taiji-r2-copy-circuit-regression-gate-v1",
        "prereg": "plans/reference/M5_R2_ARCH_COPY_CIRCUIT_PROJECT_20260925.md §4.3",
        "control_report": str(args.control),
        "treated_report": str(args.treated),
        "circuit_payload": args.circuit_payload,
        "comparability": gate,
        "judged": verdicts,
        "surface": {"control": surface_control, "treated": surface_treated},
        "surface_not_worse": surface_ok,
        "observed_not_judged": observed,
        "learn_bytes_contract": "由 pytest 面判（tests/taiji_native），本件不代判",
        "verdict": (
            "§4.3 回归门通过（A/B/C 不劣化＋表层不劣于基线）"
            if gate["comparable"] and non_degraded and surface_ok
            else "§4.3 回归门未通过/不可判（如实记录，不解释成通过）"
        ),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / "reports/taiji_r2_copy_circuit_regression_gate_20260925.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(timezone.utc).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    #: 报告写在仓库外（自检/回归跑）时也要能打印路径，不能让 relative_to 把整条门跑崩。
    shown = (
        out.relative_to(PROJECT_ROOT).as_posix() if out.is_relative_to(PROJECT_ROOT) else str(out)
    )
    print(
        json.dumps(
            {
                "comparable": gate["comparable"],
                "judged": {k: v.get("status") for k, v in verdicts.items()},
                "surface_not_worse": surface_ok,
                "verdict": report["verdict"],
                "out": shown,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
