"""A2.8-1 尾部行为诊断（零训练）：表层失血到底在"没发出来"还是"发出来又被尾巴毁掉"。

**为什么要有这一件**。§13 的成因分诊只走到答案长度就停（`min(12, 答案字节数)` 步），
所以它看的是**答案前缀**。§20 撞出一条反向事实：把寻址换成"前缀指得更准"的那一档，
前缀内发出率 17.5%→23.5%（变好），表层严格命中却 23/104→12/104（变差），
答复里出现 `明轩…明轩…明轩` 这种同词复读——**说明有一类失败发生在前缀之后，而它从未被量过**。

**这一件把失败拆成三笔，各自算法不同**（对同一批题、同一条记分链重放）：

| 类别 | 定义 | 含义 |
|---|---|---|
| `never_emitted` | 答案词在解码开头（答案长度＋2 字节内）都没完整出现 | 前缀就没做出来 ⇒ §13 那套键/续接的账 |
| `tail_destroyed` | 开头**已经**完整出现，但按表层口径（折字后整串）判**未命中** | 失血在尾巴 ⇒ 该测"为什么会锁死"，而不是继续调寻址 |
| `clean_hit` | 开头出现且表层判命中 | 真答对 |

**锚点（不过就退出码 2）**：重放得到的表层命中集合必须与已入库记分件**逐题相同**
（默认对 v1 那臂，23/104）。对不上就说明我这条重放链与记分链不同形，三笔账全是空中楼阁——
这正是 §15/§18/§20 一天里撞过三次的同一类错，这次先立规矩再读数。

**复读怎么算**：在解码字节串上找最小周期 `p`，使末尾 `3p` 个字节满足 `b[i] == b[i+p]`，
就记一次周期性尾巴（起点、周期、重复次数）。粗，但够区分"随机乱码"与"锁进循环"。

纪律：零训练（`learn=False`、不改任何持久参数）；`checkpoints/` 只读并跑前后复核 sha256；
链路只用产品原语（`SeedRuntime._serialize`／`record_told_history`）；只写 `reports/` 一份新件。
可选 `--override-addressing refit`：同一窗口下对比 §19 那档寻址覆写的尾巴形态（第一跑不做，先量基线）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

SURFACE_REPORT = PROJECT_ROOT / "reports/taiji_r2_copy_surface_extension_20260925.json"
SURFACE_MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
SEED_A_CIRCUIT = "output/taiji_r2_copy_circuit_chat/judge/circuit-final.pt"
MAX_BYTES = 64  # 与记分件同一个上限（`score_taiji_r2_copy_circuit_chat_cap.MAX_ANSWER_BYTES`）


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cut_markers(text: str) -> str:
    from api.seed_runtime import _TURN_MARKERS

    for marker in _TURN_MARKERS:
        index = text.find(marker)
        if index >= 0:
            text = text[:index]
    return text.strip()


def periodic_tail(raw: bytes) -> dict[str, Any] | None:
    """末尾是否锁进一个短周期（复读）。返回 {period, start, repeats}。"""
    length = len(raw)
    for period in range(1, 13):
        span = 3 * period
        if length < span:
            continue
        window = raw[length - span :]
        if all(window[i] == window[i + period] for i in range(span - period)):
            return {"period": period, "start": length - span, "repeats": 3}
    return None


def decode_last_turn(runtime: Any, substrate: Any, circuit: Any, turns: list[str]) -> bytes:
    """按记分链重放：复原真实历史（产品原语逐轮生成），再逐步解码最后轮。"""
    from probe_taiji_r2_a26_emission_trace import _scored_history

    from api.seed_runtime import SeedRuntime, record_told_history

    history = _scored_history(runtime, turns)
    prompt = SeedRuntime._serialize(turns[-1], history)
    record_told_history(substrate, circuit, history, episode_id="a28:final")
    config = substrate.config
    substrate.reset_dynamics(episode_id="generation")
    substrate.observe(
        int(config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    raw = bytearray()
    previous = None
    for symbol in prompt.encode("utf-8"):
        previous = int(symbol)
        step = substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)
    if previous is None or step is None:
        return b""
    #: 下一步发什么＝**`observe()` 自己返回的概率取 argmax**。
    #: 不自算读法、不猜属性、不兜底调 `generate`——§13 的链路同一性断言已经证过
    #: "手算读法与 `observe()` 给的末位概率逐位一致"，所以直接用 observe 的输出最不容易骗自己。
    for _step_index in range(MAX_BYTES):
        following = int(step.probabilities.argmax())
        raw.append(following)
        previous = following
        step = substrate.observe(following, learn=False, readout="predictive", use_memory=False)
    return bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--circuit", default=SEED_A_CIRCUIT)
    parser.add_argument(
        "--override",
        choices=("none", "refit"),
        default="none",
        help=(
            "refit＝套上 §19 那档「离线指对 94.5%」的寻址覆写，量它在**自己轨迹**上的尾巴形态"
            "（§20 的反向读数就是这么来的，基线档看不见）"
        ),
    )
    parser.add_argument("--epochs", type=int, default=2500, help="refit 档求解轮数")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from probe_taiji_r2_a26_emission_trace import extension_items
    from score_taiji_r2_copy_strict_cap import copyable_tokens

    from api.seed_runtime import SeedRuntime

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    baseline = json.loads(SURFACE_REPORT.read_text(encoding="utf-8"))
    arm = next(
        candidate
        for candidate in baseline["treated_arms"]
        if str(candidate["circuit"]) == args.circuit
    )
    recorded_hits = {str(row["id"]) for row in arm["rows"] if bool(row["hit"])}

    ids = sorted(extension_items())
    if args.limit > 0:
        ids = ids[: args.limit]
    items = extension_items()

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=4)
    substrate.copy_circuit.load_payload(
        torch.load(PROJECT_ROOT / args.circuit, weights_only=False)["copy_circuit"]
    )
    circuit = substrate.copy_circuit

    override_note = "none"
    handle = None
    if args.override == "refit":
        from price_taiji_r2_a27_address_refit import refit_m
        from probe_taiji_r2_a27_address_onpolicy import AddressingOverride

        features = PROJECT_ROOT / "output/a27_address_features.json"
        if not features.exists():
            print(json.dumps({"error": f"refit 档需要 §18 的特征件 {features.name}"}))
            return 1
        refit = refit_m(
            circuit, json.loads(features.read_text(encoding="utf-8")), args.epochs, 1e-2, 0.1
        )
        handle = AddressingOverride(refit["m"], refit["lam"] / refit["scale"])
        handle.install()
        override_note = f"refit(离线指对 {refit['refit_offline_aim']})"

    rows: dict[str, Any] = {}
    replayed_hits: set[str] = set()
    for item_id in ids:
        item = items[item_id]
        turns = [str(turn) for turn in item["turns"]]
        tokens = [str(token) for token in copyable_tokens(item)]
        raw = decode_last_turn(runtime, substrate, circuit, turns)
        text = _cut_markers(raw.decode("utf-8", errors="replace"))
        hit = any(token in text for token in tokens)
        if hit:
            replayed_hits.add(item_id)
        answer = max(tokens, key=len) if tokens else ""
        budget = len(answer.encode("utf-8")) + 2
        early = any(token.encode("utf-8") in raw[:budget] for token in tokens)
        rows[item_id] = {
            "hit_surface": bool(hit),
            "emitted_early": bool(early),
            "verdict": "clean_hit" if hit else ("tail_destroyed" if early else "never_emitted"),
            "tail": periodic_tail(raw),
            "decoded": text[:MAX_BYTES],
        }

    #: 冒烟（`--limit`）时命中集合天然不完整，锚点**不可判** ⇒ 写 null 而不是 true
    #: （把"没证"印成"证过了"是今天第三次撞同一类错，这里提前堵掉）。
    if handle is not None:
        handle.uninstall()
    anchor_ok: bool | None = None if args.limit else bool(replayed_hits == recorded_hits)
    counts: dict[str, int] = {}
    for row in rows.values():
        counts[str(row["verdict"])] = counts.get(str(row["verdict"]), 0) + 1
    looped = [row for row in rows.values() if row["verdict"] == "tail_destroyed" and row["tail"]]
    report = {
        "format": "taiji-r2-a28-tail-behavior-v1",
        "prereg": "plans/reference/SPEC-A-17_r2_a2_3b_format_align_prereg_20260925.md §20 之后",
        "chain": "scored（逐轮走记分件原语复原真实历史）＋完整 64 字节解码",
        "circuit": args.circuit,
        "addressing_override": override_note,
        "items": len(rows),
        "anchor_replays_recorded_hits": anchor_ok,  # null＝冒烟跑，不可判
        "anchor_overlap": len(replayed_hits & recorded_hits),
        "recorded_hits": len(recorded_hits),
        "verdict_counts": counts,
        "tail_destroyed_with_loop": len(looped),
        "tail_period_median": (
            sorted(int(row["tail"]["period"]) for row in looped)[len(looped) // 2]
            if looped
            else None
        ),
        "reading_rule": (
            "tail_destroyed 占未命中的多数 ⇒ 表层失血在尾巴，第一优先改测'解码尾部为什么锁死'"
            "（发射门/pooled 自反馈那一族）；never_emitted 占多数 ⇒ 回 §13 那 61 题的键/续接残余"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
        "rows": rows,
    }
    out = Path(args.out_report or "reports/taiji_r2_a28_tail_behavior_20260926.json")
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "items": report["items"],
                "anchor": anchor_ok,
                "overlap": report["anchor_overlap"],
                "verdicts": counts,
                "loops": report["tail_destroyed_with_loop"],
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
    if anchor_ok is not True or not report["base_sha256_unchanged"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
