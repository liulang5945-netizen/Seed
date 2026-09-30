"""DEBT-G17 取证：产品表层门槛①/② 的 ``well_formed`` 第一道条件在 ``str`` 上恒真 ⇒ 含 U+FFFD 的答复穿门。

为什么要有这一档（**不是新判据，是判据自证**）：`seed/surface_gate.py:51-56` 的
`well_formed` 第一条写的是 `text.encode("utf-8").decode("utf-8")`，而 `text` 是 **`str`**——
`str` 编码成 UTF-8 永远成功、再解码也永远成功，所以这一条**在类型上不可能返回 False**
（ lone surrogate 会抛 `UnicodeEncodeError`，而那个 `except` 只接 `UnicodeDecodeError`）。
这条函数被产品链用两处：`api/seed_runtime.py:444`（门槛②：坏答复不进下一轮 prompt）与
`:506`（门槛①：只回写过门答复），**两处都在 `gate_model is not None` 时生效**＝挂回路装配。

本件做三件事，全部只读、不改任何判据：
①**类型层**：证明这条 roundtrip 对含 U+FFFD 的串不抛；
②**实证层**：在**已入库**的两张表层件里数出 `well_formed=true` 且预览含 U+FFFD 的行；
③**同面重算**：用**产品随包那份 n 元工件**（`checkpoints/seed_surface_ngram.lzma`，
即 `_surface_ngram_if_armed` 武装时载入的同一个文件）对预览重跑 `surface_gate.well_formed`。

**已知边界（引用时不许掉）**：件里的 `answer` 是 `answer[:60]` 的**预览**，不是原串 ⇒
③ 的"过门"是对预览算的，比原判据更短、n 元分更低，**只会让结论偏保守**（预览都过门，原串更会过）；
预览是 `str` 切片，**不会自己造出 U+FFFD**（切字节才会）⇒ 预览里出现的 U+FFFD 来自模型输出被预算切在半个字上。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from seed import surface_gate  # noqa: E402

#: 已入库的两张表层件（同一仪器、同 104 题、只换检查点）。
ARTIFACTS = (
    PROJECT_ROOT / "reports/taiji_f0_a26p1_surface_recheck_20260930.json",
    PROJECT_ROOT / "reports/taiji_f0_a31retrain_surface_20260930.json",
)
NGRAM_ARTIFACT = PROJECT_ROOT / "checkpoints/seed_surface_ngram.lzma"


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """取件里那一条按题存 `well_formed` 的 rows；取不到就响亮失败（不许静默给 0）。"""

    for value in payload.values():
        if isinstance(value, dict) and isinstance(value.get("rows"), list):
            rows: list[dict[str, Any]] = value["rows"]
            return rows
    raise KeyError("件里没有带 rows 的臂结构，拒绝把『没找到』当成『0 条』")


def _count(path: Path, model: tuple) -> dict[str, int]:
    rows = _rows(json.loads(path.read_text(encoding="utf-8")))
    fffd = [r for r in rows if "\ufffd" in str(r.get("answer", ""))]
    wf = [r for r in rows if r.get("well_formed")]
    both = [r for r in rows if r.get("well_formed") and "\ufffd" in str(r.get("answer", ""))]
    # ③ 同面重算：用产品那份 n 元模型对预览再判一次（与件里存的位对照）
    recomputed = [r for r in both if surface_gate.well_formed(str(r.get("answer", "")), model)]
    return {
        "rows": len(rows),
        "well_formed_true": len(wf),
        "preview_contains_fffd": len(fffd),
        "well_formed_and_contains_fffd": len(both),
        "recomputed_pass_on_preview": len(recomputed),
    }


def decide(type_vacuous: bool, bypass_rows: int) -> str:
    """互斥四支——**先于数冻结**：`criterion_filters` 是本件唯一能为 false 的那一支。"""

    if not type_vacuous:
        return "not_reproduced（类型层那条 roundtrip 抛了 ⇒ 第一道条件其实是活的）"
    if bypass_rows == 0:
        return "criterion_filters（第一道条件恒真，但后三条把这些答复全拦下了）"
    if NGRAM_ARTIFACT.exists():
        return "bypass_confirmed（有答复带 U+FFFD 被 well_formed 放行；同面重算见 recomputed 列）"
    return "bypass_confirmed_rows_only（n 元工件缺席，③ 未跑；只有件里存的位可引）"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # GBK 控制台会在这里崩，而崩之前件已写盘
        sys.stdout.reconfigure(encoding="utf-8")

    # ① 类型层：三条探针，任一条抛就记 type_vacuous=false
    probes = ["中中中中中中\ufffd", "\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd", "你好"]
    try:
        for s in probes:
            s.encode("utf-8").decode("utf-8")
        type_vacuous = True
        type_error = None
    except UnicodeError as exc:  # pragma: no cover - 走到这里就是本件被否证
        type_vacuous = False
        type_error = f"{type(exc).__name__}: {exc}"

    model = surface_gate.load_surface_ngram(NGRAM_ARTIFACT) if NGRAM_ARTIFACT.exists() else None
    per_artifact = {p.name: _count(p, model) for p in ARTIFACTS} if model is not None else {}
    bypass_rows = sum(v["well_formed_and_contains_fffd"] for v in per_artifact.values())

    report = {
        "format": 1,
        "question": "产品门槛①/② 的 well_formed 第一道条件（UTF-8 合法）在 str 上是否恒真",
        "type_level": {
            "roundtrip_raises_for_fffd_string": not type_vacuous,
            "probes": [repr(s) for s in probes],
            "error": type_error,
        },
        "per_artifact": per_artifact,
        "artifact_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in ARTIFACTS
        },
        "guards": {
            "ngram_artifact_is_the_product_one": model is not None,
            "ngram_vocab_positive": bool(model and model[2] > 0),
            "both_artifacts_exist": all(p.exists() for p in ARTIFACTS),
            "rows_parsed_equal_manifest_item_count": all(
                v["rows"] == 104 for v in per_artifact.values()
            ),
        },
        "verdict": decide(type_vacuous, bypass_rows),
        "scope_note": "预览按 answer[:60] 存；③ 对预览重算 ⇒ 比原串更保守。U+FFFD 不可能由 str 切片造出。",
    }
    out = PROJECT_ROOT / "reports/taiji_a30_surface_gate_fffd_bypass_20260930.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True)[:900])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
