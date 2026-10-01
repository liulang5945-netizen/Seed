"""PLAN-A-30 §2ar：**落点收益与表层语言质量是不是此消彼长**（假说④的检验，零训练）。

**为什么要这一档**：§2ap 量到同一枚重训件在两张面上的相反读数——教师强制结束位 **96/120 胜出**，
而表层成句率从 base 的 **0.9269 掉到 0.1538**（同一 n 元判据、同尺 104 题）。§2ap-更正 又否证了我第一个
机制猜测（不是"爱铺换行"：换行数其实从 0.85 塌到 0.01，且 `well_formed` 不看换行）。
剩下的候选里有一条**可测**的：**配方把"换行之后"训成了"该结束"的位置**，于是
"越像要结束的那条答复"可能与"语言像似度越低"是同一件事的两面——若如此，
"换不换默认位"就不是免费的收益，而是**用表层质量换结束信号**。

**测法（全部复用现成链，不重抄）**：
逐题用 `score_taiji_r2_copy_circuit_chat_cap._answer_raw` 取它自己写的答复（与 §2ap 同一通道、
`learn=False`、`repetition_penalty=0.0`、同一 manifest），
- **质量侧**＝`diag_taiji_r2_surface_decode.mean_nll(stripped, ngram)`（越低越不像语料；`well_formed` 用的就是它），
- **结束信号侧**＝把同一条答复铺成 `问：{提问}\\n答：{它写的}` + `0x0A`，用 §2aa/§2ak 那件里的
  `measure()` 量这一格的 `p_boundary`／名次／是否 argmax。
两枚检查点各跑一遍（`a31` 治疗件、`a26_p1` 同血缘 base），**每臂内部各自求相关**——
base 臂是**对照**：若两臂都负，那负相关就不是配方带来的。

**判读线（先于数写死，看到数之后不许挪）**：设每臂 n＝104（Spearman ρ 的近似 SE ≈ 1/√103 ≈ 0.098）。
1. **假说④成立**：治疗臂 ρ ≤ −0.3（≈3 个 SE）**且** base 臂 ρ > −0.1；
2. **假说④否证**：两臂 |ρ| 都 < 0.1 ⇒ 这张表里看不到此消彼长，退化归"语料窄化／37.5% 中途态"，
   下一格该问的是"续满预算后表层成句回不回得来"；
3. **两臂都 ≤ −0.3** ⇒ 负相关**与配方无关**（是这台基座里"像要停"与"像语料"共变的通用性质）
   ⇒ 关于"配方造成退化"记 `not_resolved`，但要把这个共变登记成新的机理线索；
4. 其余落点 ⇒ `not_resolved`，**不许**写成"部分成立"。
**分辨率与外推禁令**：ρ 只回答"同不同向"，**不回答因果**；本件不得被引用成"配方导致表层退化"，
那需要 §2ap 末段说的对照件（同预算不带配方）。

**同件回带的两条描述性读数**（防"相关来自分母"）：每臂 `mean_nll` 的均值/中位、`well_formed` 通过率、
以及**接缝 p_boundary 与 mean_nll 的四分位交叉表**（p 高低分组 × nll 高低分组的题数）——
交叉表用来看出负相关是不是被少数极端题拉动。

**本件不动权重、不改产品默认、两枚检查点只读**：件里记两枚的 `checkpoint_sha256_before` 并在结尾复核
（`*_sha256_unchanged` 必须为真）；两臂吃**同一批题**（`items_sha256` 按 manifest 内容算，两边必须相等）。
接缝仍是本件强加的（与 §2ai/§2ak 同一句必须跟着读数走的话）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
# 本机控制台是 GBK，判读串里的 ⇒ 会崩在写完报告之后的那句 print。
sys.stdout.reconfigure(encoding="utf-8")

from diag_taiji_r2_surface_decode import (  # noqa: E402
    build_ngram_model,
    mean_nll,
    well_formed,
)
from probe_taiji_a30_ding3_stop_target_pilot import measure  # noqa: E402
from score_taiji_r2_copy_circuit_chat_cap import _answer_raw  # noqa: E402
from score_taiji_r2_copy_surface_extension import load_items  # noqa: E402

SURFACE_MANIFEST = PROJECT_ROOT / "plans" / "manifests" / "r2_copy_surface_extension_v1.json"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rank(values: list[float]) -> list[float]:
    """平均秩（并列取均值）——Spearman 需要，避免把 scipy 拉进依赖面。"""

    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else round(num / den, 4)


def quartile_table(p: list[float], nll: list[float]) -> dict[str, int]:
    """p 高/低 × nll 高/低 的题数（按各自中位分组）——看负相关是不是少数极端点拉的。"""

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0

    pm, nm = median(p), median(nll)
    out = {"": 0}
    for pi, ni in zip(p, nll, strict=True):
        key = f"p_{1 if pi >= pm else 0}_nll_{1 if ni >= nm else 0}"
        out[key] = out.get(key, 0) + 1
    del out[""]
    return out


def run_arm(
    checkpoint: Path, items: list[dict[str, Any]], *, boundary: int, mask: bool
) -> dict[str, Any]:
    from api.seed_runtime import SeedRuntime

    ngram = build_ngram_model()
    runtime = SeedRuntime.load(checkpoint)
    rows: list[dict[str, Any]] = []
    for item in items:
        turns = [str(turn) for turn in item["turns"]]
        history: list[tuple[str, str]] = []
        last_turn, last_answer = "", ""
        for index, turn in enumerate(turns):
            answer = _answer_raw(runtime, turn, history)
            prefix = f"问：{turn}\n答：{answer}".encode() + bytes([0x0A])
            face = measure(runtime, [[boundary, *prefix]], boundary, mask=mask)["faces"]["end"]
            stripped = "".join(answer.split())
            rows.append(
                {
                    "id": item["id"],
                    "family": item["family"],
                    "turn_index": index,
                    "answer_bytes": len(answer.encode()),
                    "mean_nll": round(mean_nll(stripped, ngram), 6) if stripped else None,
                    "well_formed": bool(well_formed(answer, ngram)),
                    "p_boundary": face["p_boundary"]["median"],
                    "boundary_rank": face["boundary_rank"]["median"],
                    "boundary_is_argmax": face["boundary_is_argmax_count"],
                }
            )
            last_turn, last_answer = turn, answer
            if index + 1 < len(turns):
                history.append((turn, answer))
        del last_turn, last_answer
    del runtime
    usable = [r for r in rows if r["mean_nll"] is not None and r["p_boundary"] is not None]
    p = [float(r["p_boundary"]) for r in usable]
    nll = [float(r["mean_nll"]) for r in usable]
    return {
        # ：**配对要按"这臂实际吃到的题"算指纹，不能拿同一个变量自比（那是恒真式）。
        "items_sha256": _sha(
            json.dumps([r["id"] for r in rows if r["turn_index"] == 0], ensure_ascii=False).encode()
        ),
        "texts": len(rows),
        "usable_for_correlation": len(usable),
        "well_formed_texts": sum(1 for r in rows if r["well_formed"]),
        "well_formed_rate": round(sum(1 for r in rows if r["well_formed"]) / max(len(rows), 1), 4),
        "mean_nll_mean": round(sum(nll) / len(nll), 6) if nll else None,
        "mean_nll_median": (sorted(nll)[len(nll) // 2] if nll else None),
        "p_boundary_mean": round(sum(p) / len(p), 6) if p else None,
        "seam_argmax_texts": sum(int(r["boundary_is_argmax"]) for r in usable),
        "spearman_nll_vs_pboundary": spearman(nll, p),
        "quartile_cross_table_p_high_nll_high": quartile_table(p, nll),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", default="output/a31_ding3_boundary/checkpoint.pt")
    parser.add_argument("--base", default="output/a26_p1/checkpoint.pt")
    parser.add_argument("--manifest", default=str(SURFACE_MANIFEST))
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 题（冒烟用；0＝全量）")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    from taiji import TaijiConfig

    boundary = int(TaijiConfig().boundary_symbol)
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    raw_sha = _sha(manifest.read_bytes())
    items = load_items(manifest)
    if args.limit:
        items = items[: args.limit]
    items_sha = _sha(
        json.dumps([i["id"] for i in items], ensure_ascii=False).encode()
    )  # ：**与 run_arm 里按行算的那把同一表达式**，否则配对守卫会比出假阴性（冒烟实测就是这样）

    retrain = PROJECT_ROOT / args.retrain
    base = PROJECT_ROOT / args.base
    sha_before = {"retrain": _sha(retrain.read_bytes()), "base": _sha(base.read_bytes())}

    runs = {}
    for label, ckpt in (("retrain", retrain), ("base", base)):
        runs[label] = run_arm(ckpt, items, boundary=boundary, mask=args.mask)
        runs[label]["checkpoint_sha256_before"] = sha_before[label]

    tr = runs["retrain"]["spearman_nll_vs_pboundary"]
    bs = runs["base"]["spearman_nll_vs_pboundary"]
    if tr is None or bs is None:
        verdict = "not_resolved（相关算不出来：可用样本 <3 或某一列全等）"
    elif tr <= -0.3 and bs > -0.1:
        verdict = (
            f"假说④成立：治疗臂 ρ={tr}（≤−0.3）而 base 臂 ρ={bs}（>−0.1）"
            "⇒ 在这枚重训件上，越像要结束的答复越不像语料；这是换默认位那笔的**真代价**（不是因果证明，因果需对照件）"
        )
    elif abs(tr) < 0.1 and abs(bs) < 0.1:
        verdict = (
            f"假说④否证：两臂 |ρ| 都 <0.1（治疗 {tr}／base {bs}）"
            "⇒ 看不到此消彼长；表层退化归语料窄化／37.5% 中途态，下一格该问续满预算后成句回不回得来"
        )
    elif tr <= -0.3 and bs <= -0.3:
        verdict = (
            f"两臂都 ≤−0.3（治疗 {tr}／base {bs}）⇒ 负相关与配方无关，是这台基座里「像要停」与「像语料」"
            "共变的通用性质；关于配方贡献记 not_resolved，但这个共变要登记成新机理线索"
        )
    else:
        verdict = f"not_resolved（ρ 落在预注册线之间：治疗 {tr}／base {bs}）"

    report = {
        "format": "taiji-a30-recipe-surface-tradeoff-v1",
        "prereg": "PLAN-A-30 §2ar（判读线先于数写在件 docstring）",
        "question": "落点配方的结束信号收益与表层语言质量是否此消彼长（逐题 p_boundary 与 mean_nll 的 Spearman）",
        "manifest": manifest.name,
        "manifest_sha256": raw_sha,
        "items_sha256": items_sha,
        "items": len(items),
        "mask": bool(args.mask),
        "decision_face": "utf8_masked_legal_set" if args.mask else "full_alphabet",
        "runs": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in runs.items()},
        "rows": {k: v["rows"] for k, v in runs.items()},
        "verdict": verdict,
        "instrument_guard": {
            "same_items_both_arms": (
                runs["retrain"]["items_sha256"] == runs["base"]["items_sha256"] == items_sha
            ),
            "both_arms_same_text_count": runs["retrain"]["texts"] == runs["base"]["texts"],
            "retrain_sha256_unchanged": _sha(retrain.read_bytes()) == sha_before["retrain"],
            "base_sha256_unchanged": _sha(base.read_bytes()) == sha_before["base"],
        },
        "operational_definition": (
            "接缝由本件强加（问：q\\n答：模型自己写的答案 + 一个换行）；质量侧用与 §2ap 同一个 n 元判据的"
            "mean_nll（越低越不像语料）。相关只回答同不同向，不回答因果。"
        ),
        "started_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out = (
        Path(args.out_report)
        if args.out_report
        else PROJECT_ROOT
        / f"reports/taiji_a30_recipe_surface_tradeoff_{len(items)}item_20260930.json"
    )
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        json.dumps(
            {
                "items": len(items),
                "retrain": {
                    k: runs["retrain"][k]
                    for k in (
                        "well_formed_rate",
                        "mean_nll_mean",
                        "spearman_nll_vs_pboundary",
                        "seam_argmax_texts",
                    )
                },
                "base": {
                    k: runs["base"][k]
                    for k in (
                        "well_formed_rate",
                        "mean_nll_mean",
                        "spearman_nll_vs_pboundary",
                        "seam_argmax_texts",
                    )
                },
                "verdict": verdict,
                "guard": report["instrument_guard"],
                "out": out.name,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
