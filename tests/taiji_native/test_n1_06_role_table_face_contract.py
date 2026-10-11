"""`audit_taiji_n1_03_role_table_face.py` 的契约册：三条判据正反两支都要跑，且钉住今天的实名清单。

这台仪器的危险之处不在"算错"，而在**取法自己造出假通过**——我在一格里就犯过三次：
①只看"所有实名都能解析"（判据原文是"至少一枚"），会把有承载的环报成缺承载；
②反过来"任一末段命中即算解析"，会让 `train_seed_corpus.py` 靠末段 `py`、`substrate.state.*`
靠**空末段**命中——那是任何输入都能过的恒真式；
③把"信封由跑道器写"的两枚自述键判成"未取到"，因为我只扫了 `taiji/`＋`seed/`。
所以这册把每一条都钉成"改坏了就会红"。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_n1_03_role_table_face.py"

MINI_PREREG = """# mini

| 环 | 原文 | 承载 | 件 |
|---|---|---|---|
| ① 第一环 | a | `taiji/memory.py` 与 `motor_context` | x |
| ② 第二环 | b | `substrate.state.*` | x |
| ③ 第三环 | c | `totally_made_up_anchor_xyz` | x |
| ④ 第四环 | d | `taiji.kernel.state.motor_context` | x |
| ⑤ 第五环 | e | `answer_source` | x |

| 角色 | 生产者 | 实际消费者 | 更新来源 | 恢复归属 |
|---|---|---|---|---|
| Q | p | c | u | `self_key` |
| E | p | c | u | `config` |
| M | p | — | u | `config` |
| W | p | c | u | `config` |
| R | p | c | u | `config` |
"""


def _load():
    spec = importlib.util.spec_from_file_location("n1_05_role_table_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _root(tmp_path: Path, extra: dict[str, str] | None = None) -> Path:
    for rel, text in {
        "taiji/memory.py": (
            "class EpisodicField:\n    motor_context = 1\n\n\n"
            'def build_envelope():\n    return {"config": {}}\n'
        ),
        "scripts/training/train_seed_corpus.py": (
            'def save(envelope):\n    envelope["self_key"] = 1\n    return envelope\n'
        ),
    }.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    for rel, text in (extra or {}).items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    return tmp_path


def test_empty_tables_fail_closed(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path)
    try:
        module.judge("# nothing here\n", root)
    except ValueError as exc:
        assert "什么都没测到" in str(exc), exc
    else:
        raise AssertionError("表格解析为空必须响亮失败，不许读成「表是完整的」")


def test_ring_rule_is_at_least_one_strong_anchor(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path)
    report = module.judge(MINI_PREREG, root)
    j1 = report["j_n1c_1"]
    by_ring = {row["ring"]: row for row in j1["rings"]}
    assert by_ring["① 第一环"]["strongly_resolved"] >= 1, by_ring["① 第一环"]
    assert by_ring["③ 第三环"]["strongly_resolved"] == 0, by_ring["③ 第三环"]
    assert "③ 第三环" in j1["rings_without_evidence"], j1["rings_without_evidence"]
    assert j1["verdict"] == "chain_incomplete", j1


def test_weak_and_dead_anchors_are_not_counted_as_carriers(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path)
    j1 = module.judge(MINI_PREREG, root)["j_n1c_1"]
    weak = " ".join(j1["anchors_with_weak_resolution_only"])
    dead = " ".join(j1["anchors_that_do_not_resolve"])
    assert "taiji.kernel.state.motor_context" in weak, j1
    assert "substrate.state.*" in dead, j1
    by_ring = {row["ring"]: row for row in j1["rings"]}
    assert by_ring["② 第二环"]["strongly_resolved"] == 0, by_ring["② 第二环"]
    assert "② 第二环" in j1["rings_without_evidence"], j1


def test_every_real_file_name_anchor_resolves_strongly(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path)
    blob = "\n".join(
        (root / rel).read_text(encoding="utf-8") for rel in module.tracked_files(root, "*.py")
    )
    tracked = module.tracked_files(root, "*.py")
    resolved = module.resolve_name(root, "train_seed_corpus.py", blob, "x", tracked)
    assert resolved["strong"] is True and "as_tracked_basename" in resolved["kinds"], resolved


def test_empty_cell_is_reported_by_role_and_column(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path)
    j2 = module.judge(MINI_PREREG, root)["j_n1c_2"]
    assert j2["verdict"] == "table_incomplete", j2
    assert j2["empty_cells"] == ["M/实际消费者"], j2


def test_envelope_key_written_only_by_runner_still_verifies(tmp_path: Path) -> None:
    """scope 修复的钉子：信封由**跑道器**落盘，只扫 `taiji/`＋`seed/` 会把它误判成"未取到"。"""

    module = _load()
    root = _root(tmp_path)
    j4 = module.judge(MINI_PREREG, root)["j_n1c_4"]
    checks = {row["key"]: row for row in j4["checks"]}
    assert checks["self_key"]["verifiable"] is True, checks["self_key"]
    assert checks["self_key"]["written_into_envelope_by"] == [
        "scripts/training/train_seed_corpus.py"
    ], checks["self_key"]
    assert j4["verdict"] == "recovery_claims_verifiable", j4
    assert j4["write_side_scan_faces"] == ["taiji", "seed", "scripts/training"], j4


def test_missing_envelope_key_is_named_not_silently_passed(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path)
    prereg = MINI_PREREG.replace("`self_key`", "`never_written_anywhere_key`")
    j4 = module.judge(prereg, root)["j_n1c_4"]
    assert j4["verdict"] == "unverifiable_recovery_claim", j4
    assert j4["unverifiable"] == ["never_written_anywhere_key"], j4


def test_real_repository_face_reproduces_today_reading() -> None:
    """真面锚点：今天的 25 枚实名分解与三枚待修实名必须逐值复现（改了预注册件就要同步改这里）。"""

    module = _load()
    report = module.build_report(PROJECT_ROOT)
    j1 = report["j_n1c_1"]
    assert j1["verdict"] == "chain_complete", j1
    assert j1["rings_parsed"] == 5 and j1["rings_without_evidence"] == [], j1
    assert report["j_n1c_2"]["verdict"] == "table_registered", report["j_n1c_2"]
    assert report["j_n1c_2"]["cells_parsed"] == 20, report["j_n1c_2"]
    assert report["j_n1c_4"]["verdict"] == "recovery_claims_verifiable", report["j_n1c_4"]
    assert j1["anchors_that_do_not_resolve"] == ["substrate.state.*"], j1
    assert len(j1["anchors_with_weak_resolution_only"]) == 2, j1
    assert any(
        name.startswith("taiji.kernel.state.motor_context")
        for name in j1["anchors_with_weak_resolution_only"]
    ), j1
    total_anchors = sum(row["named_anchors"] for row in j1["rings"])
    assert total_anchors == 25, total_anchors
    assert len(report["what_this_does_not_prove"]) >= 3, report["what_this_does_not_prove"]
    assert json.dumps(report, ensure_ascii=False)
