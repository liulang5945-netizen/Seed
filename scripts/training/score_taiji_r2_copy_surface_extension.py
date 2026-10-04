"""§4.3a 表层子判据的**扩展分母**跑法（题集 `r2_copy_surface_extension_v1.json`，104 题≈260 条文本）。

预注册：`plans/reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md`。

**这条仪器只为解决一件事**：原 §4.3a 的表层判据在 36 题（≈72 条文本）上**结构性不可判**——
裁定要求"成句差 ≥3 且两次独立取数同向"，而那个分母上 ±1 条≈2.8pp，等于要求 8% 的摆动。
本件把分母抬到约 260 条文本（±1 条≈0.4pp），并把"两次独立取数"落到**两个独立初始化的电路**上
（seed-A＝A2.3b 已落地的那份；seed-B＝`--circuit-seed 20260926` 重训的那份）。

链路：与 A2.3b 的 chat CAP 完全同一条（`SeedRuntime._serialize` 铺文本、`_record_told_history` 入库、
`generate_input(reset=True)` 取基底原始字节、`_TURN_MARKERS` 折字、n 元口径成句判定）——
**换分母不换链**，否则读数不可比（本仓多次实测：换链路会让同一判据反向）。

判读（冻结在预注册里，这里只实现）：
* 每个电路各自与对照臂比成句文本数；
* **两个电路同向**且成句差 ≥3 条 ⇒ 判"表层劣化/改善"；
* 否则一律 `not_resolved`（既不判劣化也不判无代价）。
严格真命中与可解码率一并报出，但**不参与**这条门的判定。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v1.json"
MIN_GAP_TEXTS = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_items(path: Path = MANIFEST) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["dimensions"]["X"]["items"])


def build_circuit_carried_envelope(
    base: Path, circuit: str, out_path: Path, *, max_events: int = 4
) -> dict[str, Any]:
    """把已训电路装进**产品信封**（受检基底只读；写盘只落到 `out_path`）。

    为什么要这一档：产品挂载复制回路的**唯一**入口是"档里带回路 ⇒ 恢复时自动挂载"
    （`SeedRuntime.load` → `Seed.from_checkpoint` → `Taiji.restore`），而 `enable_copy_circuit`
    是探针/评测专用的显式 opt-in。只量后者，就永远不知道"电路随基底出厂"那一天产品面读到什么
    ——包括裁定 (b) 的证据门在那条路上有没有真的开。
    """
    import torch

    from seed import Seed

    model = Seed.from_checkpoint(torch.load(base, map_location="cpu", weights_only=True))
    substrate = model.substrate
    if substrate.copy_circuit is None:
        substrate.mount_copy_circuit(max_events=max_events)
    payload = torch.load(PROJECT_ROOT / circuit, map_location="cpu", weights_only=False)[
        "copy_circuit"
    ]
    substrate.copy_circuit.load_payload(payload)
    envelope = model.checkpoint()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(envelope, out_path)
    return {
        "base": base.relative_to(PROJECT_ROOT).as_posix(),
        "base_sha256": _sha256(base),
        "circuit": circuit,
        "envelope": (
            out_path.relative_to(PROJECT_ROOT).as_posix()
            if out_path.is_relative_to(PROJECT_ROOT)
            else out_path.as_posix()
        ),
        "envelope_bytes": out_path.stat().st_size,
        "envelope_sha256": _sha256(out_path),
    }


def run_arm(
    items: list[dict[str, Any]],
    checkpoint: Path,
    circuit: str | None,
    *,
    evidence_utf8_gate: bool = False,
    close_gate_after_load: bool = False,
    surface: bool = False,
    window_steps: int | None = None,
    product_window_steps: int | None = None,
    injection_mode: str | None = None,
) -> dict[str, Any]:
    """一臂：跑完 104 题，按产品 chat 协议取基底原始答复，统计表层三率。

    `evidence_utf8_gate`（PLAN-A-25）：只在评测期把复制回路的加性证据按 UTF-8 位置状态门控
    ——**默认 False ⇒ 与冻结链逐位相同**；开启走 `Taiji.set_copy_evidence_utf8_gate` 的运行时覆写
    （不改 config、不进 payload，所以"唯一变量＝门开/关"这条归因干净）。

    `circuit=None` 时**不再等于"没有回路"**：档里带回路则产品入口自动挂载
    （`mount_entry="envelope_auto_mount"`）；那种情况下裁定 (b) 已把门开成 TRUE，
    `close_gate_after_load=True` 是显式把它关回去的对照档（＝修法之前的产品读数）。
    """

    from eval_taiji_r2_readout_retrain import build_ngram_model, well_formed
    from score_taiji_r2_copy_circuit_chat_cap import _answer_raw

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    substrate = runtime.model.substrate
    mount_entry = "none"
    if circuit:
        runtime.enable_copy_circuit(PROJECT_ROOT / circuit)
        mount_entry = "enable_copy_circuit"
        if evidence_utf8_gate:
            substrate.set_copy_evidence_utf8_gate(True)
    elif substrate.copy_circuit is not None:
        mount_entry = "envelope_auto_mount"
        if close_gate_after_load:
            substrate.set_copy_evidence_utf8_gate(False)
    #: 产品档（与 L2 探针 v20/v21、cap 仪器同名旗标）：门开在产品代码里，
    #: 计步基与每趟复位都在 `Taiji.generate()` 内 ⇒ 这台仪器不需要自己数步。
    if product_window_steps is not None:
        if window_steps is not None:
            raise SystemExit("产品档与替身档不能同开——同开就分不出读数来自哪条路径")
        if product_window_steps <= 0:
            raise SystemExit("产品档的 K 必须是正整数")
        substrate.set_copy_evidence_window_steps(product_window_steps)
    #: SPEC-A-26 形状甲（与 L2 探针 v40、cap 仪器同名旗标）：注入形状走产品原生开关；
    #: `None` ⇒ 根本不调用 ⇒ 逐位不变。本臂没有回路时给了档属错配，响亮停下。
    if injection_mode is not None:
        if substrate.copy_circuit is None:
            raise SystemExit("要求注入形状档但本臂没有回路 ⇒ 竞争式没有可竞争的通道")
        substrate.set_copy_evidence_injection_mode(injection_mode)
    #: DEBT-G30 顺手迁移（本轮加旗标即本仪器的"下次升版"场合）：有效值读产品公开出口，
    #: 不再扒私有字段重推（那条式子只住产品一处）。
    gate_effective = bool(substrate.copy_evidence_utf8_gate_state()["effective"])
    ngram = build_ngram_model()
    #: 第四十九次停靠：**发射时序档**，与 L2／cap 两台仪器共用 `_make_window_armed_evidence`，
    #: 步刻度也用同一个 `generation_loop_span`（1 步＝喂进 1 字节），换一轮生成即清零。
    window_counters = [0, 0, 0]
    loop_steps = [0]
    if window_steps is not None:
        import sys

        from probe_taiji_a30_copy_evidence_dose import _make_window_armed_evidence
        from probe_taiji_a30_stop_failure import generation_loop_span

        if substrate.copy_circuit is None:
            raise RuntimeError("要求窗口档但回路不在场 ⇒ 没有可门控的证据通道")
        wf_first, wf_last = generation_loop_span(type(substrate).generate)
        w_original_observe = substrate.observe

        def w_observing(symbol: Any, **kwargs: Any) -> Any:
            frame = sys._getframe(1)
            step = w_original_observe(symbol, **kwargs)
            if frame.f_code.co_name == "generate" and wf_first <= frame.f_lineno <= wf_last:
                loop_steps[0] += 1
            return step

        substrate.observe = w_observing  # type: ignore[method-assign]
        armed_window, window_counters = _make_window_armed_evidence(
            substrate.copy_circuit.evidence, int(window_steps), lambda: loop_steps[0]
        )
        substrate.copy_circuit.evidence = armed_window  # type: ignore[method-assign]

    texts: list[str] = []
    hits = 0
    rows: list[dict[str, Any]] = []
    for item in items:
        history: list[tuple[str, str]] = []
        turns = [str(turn) for turn in item["turns"]]
        answer = ""
        for index, turn in enumerate(turns):
            loop_steps[0] = 0  # 换一轮生成：窗口步刻度从 0 重数
            answer = (
                # 显式 `repetition_penalty=0.0`：产品默认 2026-09-28 起是 2.0（owner 裁定，
                # PLAN-A-30 §2f），而本件的表层链读数是在旧默认位上取的——钉住才复现得动。
                runtime.chat(turn, history=history, learn=False, repetition_penalty=0.0)
                if surface
                else _answer_raw(runtime, turn, history)
            )
            texts.append(answer)
            if index + 1 < len(turns):
                history.append((turn, answer))
        tokens = [str(token) for token in item["expected_contains"]]
        hit = any(token in answer for token in tokens)
        hits += int(hit)
        rows.append(
            {
                "id": item["id"],
                "family": item["family"],
                "hit": hit,
                "answer": answer[:60],
                "well_formed": bool(well_formed(answer, ngram)),
            }
        )
    decodable = sum(1 for text in texts if "\ufffd" not in text)
    formed = sum(1 for text in texts if well_formed(text, ngram))
    #: **切尾感知**的诊断列（不改判定）：固定字节预算会把"缓冲切在字中间"也算成不可解码。
    #: 去掉末尾连续的 `\ufffd` 后仍含 `\ufffd` 才算**真的吐过非法字节**。
    #: 本仓已在两处独立踩过同一坑（F0 的 ON 臂 64 字节、本件 24 字节）——
    #: 只报 `utf8_decodable_rate` 会把度量缺陷读成模型缺陷。
    trimmed_clean = sum(1 for text in texts if "\ufffd" not in text.rstrip("\ufffd"))
    return {
        "circuit": circuit,
        #: 两条面**不许互换**（本仓裁定：同一份读数两条量）：原始字节链＝基底直接吐出的字节；
        #: 表层链＝`SeedRuntime.chat()` 过语言器官后的文本（带 SPEC-R2-02 的 utf8_strict 掩码）。
        "chain": "product_surface_chat" if surface else "base_raw_bytes",
        #: 挂回路走的是哪条入口，必须落在件上：`enable_copy_circuit` 是探针/评测的显式 opt-in，
        #: `envelope_auto_mount` 才是产品自己那条路（裁定 (b) 的证据门此前只在前者生效）。
        "mount_entry": mount_entry,
        #: 窗口档自述：K 与"发出／静音"两侧计数（两侧都非零才算这档真的在窗内发过、窗外拦过）。
        "evidence_window_steps": window_steps,
            "product_window_steps": product_window_steps,
        "window_arm": {
            "calls": window_counters[0],
            "emitted": window_counters[1],
            "silenced": window_counters[2],
        },
        "gate_effective": gate_effective,
        #: SPEC-A-26 形状甲自述：本臂的请求档与产品自述（生效档＋competitive 应用步数）。
        "injection_mode": injection_mode,
        "copy_evidence_injection_state": substrate.copy_evidence_injection_state(),
        "items": len(rows),
        "texts": len(texts),
        "well_formed_texts": formed,
        "well_formed_rate": round(formed / max(len(texts), 1), 4),
        "utf8_decodable_rate": round(decodable / max(len(texts), 1), 4),
        "utf8_decodable_trimmed_rate": round(trimmed_clean / max(len(texts), 1), 4),
        "illegal_after_tail_trim_rate": round(1.0 - trimmed_clean / max(len(texts), 1), 4),
        "strict_hits": hits,
        "rows": rows,
            "product_window_stats": substrate.copy_evidence_window_stats(),
    }


def rule_verdict(control: dict[str, Any], treated: list[dict[str, Any]]) -> dict[str, Any]:
    """判定：每个电路与对照比成句**文本条数**；两个电路同向且差 ≥3 条才下结论。"""
    per_circuit = []
    directions = []
    for arm in treated:
        gap = arm["well_formed_texts"] - control["well_formed_texts"]
        direction = "worse" if gap < 0 else ("better" if gap > 0 else "flat")
        directions.append(direction)
        per_circuit.append(
            {
                "circuit": arm["circuit"],
                "gap_texts": gap,
                "direction": direction,
                "meets_min_gap": abs(gap) >= MIN_GAP_TEXTS,
                "control_rate": control["well_formed_rate"],
                "treated_rate": arm["well_formed_rate"],
            }
        )
    worse = [
        row for row, direction in zip(per_circuit, directions, strict=False) if direction == "worse"
    ]
    better = [
        row
        for row, direction in zip(per_circuit, directions, strict=False)
        if direction == "better"
    ]
    if len(per_circuit) < 2:
        # "两电路同向"在只有一个电路时会**空洞地成立**（len(worse)==len(per_circuit)==1），
        # 那就等于用一次取数下判据——正是 §3 禁止的事，这里显式堵住。
        return {
            "min_gap_texts": MIN_GAP_TEXTS,
            "per_circuit": per_circuit,
            "status": "not_resolved",
            "verdict": f"not_resolved（独立取数只有 {len(per_circuit)} 次，判据要求 ≥2 次）",
        }
    if len(worse) == len(per_circuit) and all(row["meets_min_gap"] for row in worse):
        verdict = "表层劣化成立（两电路同向、每条差 ≥3）"
        status = "degraded"
    elif len(better) == len(per_circuit) and all(row["meets_min_gap"] for row in better):
        verdict = "表层改善（两电路同向、每条差 ≥3）"
        status = "improved"
    else:
        verdict = f"not_resolved（未同时满足『两电路同向』与『差 ≥{MIN_GAP_TEXTS} 条』）"
        status = "not_resolved"
    return {
        "min_gap_texts": MIN_GAP_TEXTS,
        "per_circuit": per_circuit,
        "status": status,
        "verdict": verdict,
    }


def _window_flag_honored(requested, arms) -> bool:
    """给了 `--product-window-steps` 就必须有至少一臂带上它，否则这次读数不是「门开了」的对照。

    实测成因（2026-10-03）：控制臂当时不接这个旗标，而治疗臂列表在没给 `--circuit` 时为空 ⇒
    两件的读数**逐字节相同**、全文找不到 `128`。静默空转的旗标比不给更坏，它会产出一条假对照。
    """

    if requested is None:
        return True
    return any(arm.get("product_window_steps") == requested for arm in arms)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument(
        "--circuit",
        action="append",
        default=[],
        help="治疗臂电路 payload，可给多次（多次＝多次独立取数）",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="题集 JSON（默认 v1 软干扰）。给 v2 时读数必须与 v1 **分开报**，"
        "合并总分就是把两个难度档摊平成一个（SPEC-A-21 §6、SPEC-A-22 §3）。",
    )
    parser.add_argument("--out-report", default=None)
    parser.add_argument(
        "--copy-evidence-utf8-gate",
        action="store_true",
        help="PLAN-A-25：把复制回路的加性证据按 UTF-8 位置状态门控（默认关 ⇒ 与冻结链逐位相同）",
    )
    parser.add_argument(
        "--circuit-in-envelope",
        action="store_true",
        help="PLAN-A-28：把 --circuit 的已训回路装进产品信封，再走 SeedRuntime.load 的**自动挂载**"
        "档取数（产品自己唯一能挂回路的路径）；受检基底仍只读",
    )
    parser.add_argument(
        "--auto-mount-gate-closed",
        action="store_true",
        help="在自动挂载档上再补一臂把门显式关回去＝裁定 (b) 未补到产品入口时产品会读到的数",
    )
    parser.add_argument(
        "--envelope-dir",
        default="output/a28_product_face",
        help="产品信封落点（默认 output/ 下的具名目录；不写 checkpoints/）",
    )
    parser.add_argument(
        "--surface-chain",
        action="store_true",
        help="PLAN-A-28 §8 的欠账：所有档改走 `SeedRuntime.chat()`（产品表层链，过语言器官＋"
        "SPEC-R2-02 掩码），与原始字节链**分开报**——两条量不许互换",
    )
    parser.add_argument(
        "--evidence-window-steps",
        type=int,
        default=None,
        help="第四十九次停靠：只在答复的前 K 步发复制回路证据，之后静音（与 L2/cap 共用同一副档与同一把步刻度）。"
        "默认关 ⇒ 与冻结链逐位相同。",
    )
    parser.add_argument(
        "--product-window-steps",
        type=int,
        default=None,
        help="产品档：调产品侧原生生命周期门（`Taiji.set_copy_evidence_window_steps`），不是本仪器的替身档。"
        "默认 None ⇒ 逐位不变；与 --evidence-window-steps 互斥。",
    )
    parser.add_argument(
        "--copy-evidence-injection-mode",
        type=str,
        default=None,
        choices=["additive", "competitive"],
        help="SPEC-A-26 形状甲（作用于每枚治疗臂）：证据注入形状——additive＝现行裸加，"
        "competitive＝独立候选头同格竞争（不裸加）。调产品原生开关，不给 ⇒ 不调用 ⇒ 逐位不变。",
    )
    args = parser.parse_args()
    surface = bool(args.surface_chain)

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    manifest = MANIFEST if not args.manifest else Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    items = load_items(manifest)
    control = run_arm(
        items, checkpoint, None, surface=surface, product_window_steps=args.product_window_steps
    )
    treated = [
        run_arm(
            items,
            checkpoint,
            circuit,
            evidence_utf8_gate=bool(args.copy_evidence_utf8_gate),
            surface=surface,
            window_steps=args.evidence_window_steps,
            product_window_steps=args.product_window_steps,
            injection_mode=args.copy_evidence_injection_mode,
        )
        for circuit in args.circuit
    ]
    #: PLAN-A-28 档：同一份回路、同一个基底，只换"怎么挂上来"。
    #: `enable_copy_circuit` 与 `envelope_auto_mount` 若逐位相同 ⇒ 产品入口与探针入口等价；
    #: 门关档则是裁定 (b) 没补到产品入口时产品会读到的数（合法性代价直接可见）。
    if not _window_flag_honored(args.product_window_steps, [control, *treated]):
        raise SystemExit(
            "--product-window-steps 给了 " + str(args.product_window_steps) +
            "，但没有任何一臂带上它（控制臂不接旗标、治疗臂列表为空）⇒ 这次读数不是门开了的对照，"
            "不落件；要么给 --circuit，要么让控制臂接旗标。"
        )

    envelope_meta: list[dict[str, Any]] = []
    if args.circuit_in_envelope:
        envelope_dir = Path(args.envelope_dir)
        if not envelope_dir.is_absolute():
            envelope_dir = PROJECT_ROOT / envelope_dir
        for index, circuit in enumerate(args.circuit):
            env_path = envelope_dir / f"{checkpoint.stem}_with_circuit_{index}.pt"
            meta = build_circuit_carried_envelope(checkpoint, circuit, env_path)
            meta["auto_mount_gate_effective"] = True
            envelope_meta.append(meta)
            treated.append(run_arm(items, env_path, None, surface=surface))
            if args.auto_mount_gate_closed:
                treated.append(
                    run_arm(items, env_path, None, close_gate_after_load=True, surface=surface)
                )
    report = {
        "format": "taiji-r2-copy-surface-extension-v1",
        "prereg": "plans/reference/SPEC-A-21_r2_surface_extension_prereg_20260925.md",
        #: 本轮（A2.5）的判读线与三档归因矩阵钉在 SPEC-A-22 §3/§9；仪器本身仍是 §A-21 那台。
        "judge_prereg": "plans/reference/SPEC-A-22_r2_a2_5_query_conditioned_selector_prereg_20260926.md",
        "manifest": (
            manifest.relative_to(PROJECT_ROOT).as_posix()
            if manifest.is_relative_to(PROJECT_ROOT)
            else manifest.as_posix()
        ),
        #: 文件名容易看不出来，这一列是"哪条链"的唯一自证（两条面不许互换）。
        "chain": "product_surface_chat" if surface else "base_raw_bytes",
        "manifest_sha256": _sha256(manifest),
        "checkpoint": args.checkpoint,
        #: DEBT-G21：表层这台的信封此前只有 `manifest_sha256` 与 `base_sha256_unchanged`，
        #: 底座是按**路径**认的；跨工件配对（§第五十次停靠）因此只能比路径。补上跑前那一次读盘的哈希。
        "checkpoint_sha256": sha_before[:16],
        #: PLAN-A-25：门开/关必须落在件上，否则两份读数看起来像同一次实验。
        "copy_evidence_utf8_gate": bool(args.copy_evidence_utf8_gate),
        #: SPEC-A-26 形状甲：请求档（作用于每枚治疗臂；控制臂不接）与"被走到"守卫——
        #: 每枚治疗臂的生效档必须等于请求，competitive 档的 competitive_steps 必须 >0。
        "copy_evidence_injection_mode": args.copy_evidence_injection_mode,
        "injection_mode_honored": (
            args.copy_evidence_injection_mode is None
            or (
                bool(treated)
                and all(
                    arm.get("copy_evidence_injection_state", {}).get("mode")
                    == args.copy_evidence_injection_mode
                    for arm in treated
                )
                and (
                    args.copy_evidence_injection_mode == "additive"
                    or all(
                        arm.get("copy_evidence_injection_state", {}).get("competitive_steps", 0) > 0
                        for arm in treated
                    )
                )
            )
        ),
        "circuit_carried_envelopes": envelope_meta,
        "control_no_circuit": control,
        "treated_arms": treated,
        "surface_verdict": rule_verdict(control, treated),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    #: 默认名跟着题集走——v2 的读数撞进 v1 那个文件名，比多打一个参数贵得多
    #: （§3 要求两集**分开报**，文件名是最容易被误读的那一层）。
    manifest_tag = manifest.stem.rsplit("_", 1)[-1]  # ..._v1 / ..._v2
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT / f"reports/taiji_r2_copy_surface_extension_{manifest_tag}_20260925.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "texts_per_arm": control["texts"],
                "control_wf": control["well_formed_texts"],
                "treated": [
                    (
                        arm["mount_entry"],
                        bool(arm["gate_effective"]),
                        arm["strict_hits"],
                        arm["well_formed_texts"],
                        arm["utf8_decodable_trimmed_rate"],
                    )
                    for arm in treated
                ],
                "status": report["surface_verdict"]["status"],
                "base_unchanged": report["base_sha256_unchanged"],
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else out.as_posix()
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
