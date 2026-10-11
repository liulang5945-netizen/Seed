"""`audit_taiji_default_reachability.py` 的契约册：默认位可达性由"我手登记"变成可复算，且每一路都能为假。

这册要钉住的四件事
------------------
① **判定权在两张面**：只有 `product`（`taiji/`＋`seed/`）与 `runner`（跑道器那一份文件）的调用者
   参与"默认装配可达"；测试与仪器（`eval_*`／`probe_*`／`audit_*`…，乃至测试断言）不参与——
   把它们算进来就等于宣布"有人在默认装配下走到"，而那正是 ㊵-699／DEBT-G106 付过学费的形状。
② **旗标门控的两支都要跑**：同一个函数，调用者从"裸调"改成"包在 `if args.<store_true 旗标>:` 里"
   必须让判级从 `reachable_at_default` 翻成 `flag_gated`；不翻就是恒真式（我判过这种"看起来会红"的门）。
③ **门控识别不靠猜**：只认同文件 `argparse` 里 `action="store_true"` 或 `default=False` 的目标名；
   一个没有任何旗标的文件不许产生 `flag_gated`。
④ **与已发表读数对齐**：真面上必须复现 ㊵-699 那条——`taiji/model.py` 里
   `propose_adaptive_residual_growth_candidate` 这个读取点判 `flag_gated`，且门控旗标是 `n5_shadow`。

夹具都要 `git add`（扫描面按版本控制取，见 [[guard-scan-face-git-not-disk]]）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_default_reachability.py"
WRITER_REPORT = PROJECT_ROOT / "reports/taiji_write_only_key_face_20261011.json"


def _load():
    spec = importlib.util.spec_from_file_location("n1_05_default_reachability", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    return tmp_path


_PRODUCT = (
    "class Holder:\n"
    "    def produce(self, value):\n"
    "        self.motor_context = value\n\n\n"
    "def consume(holder):\n"
    "    if holder.motor_context:\n"
    "        return 1\n"
    "    return 0\n"
)


def test_ungated_product_caller_is_reachable_at_default(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": _PRODUCT,
            "taiji/caller.py": (
                "from taiji.prod import consume\n\n\ndef run(holder):\n    return consume(holder)\n"
            ),
        },
    )
    rows = module.classify(root, [{"file": "taiji/prod.py", "line": 6, "function": "consume"}])
    assert rows[0]["verdict"] == "reachable_at_default", rows
    assert rows[0]["product_ungated_callers"] >= 1, rows


def test_flag_gated_and_ungated_callers_flip_the_verdict(tmp_path: Path) -> None:
    """同一段读取点：调用者包进门控 ⇒ flag_gated；去掉门 ⇒ reachable_at_default。"""

    module = _load()
    gated_root = _fixture(
        tmp_path / "gated",
        {
            "taiji/prod.py": _PRODUCT,
            "scripts/training/train_seed_corpus.py": (
                "import argparse\n\n\n"
                "def build_parser():\n"
                "    parser = argparse.ArgumentParser()\n"
                '    parser.add_argument("--n5-shadow", action="store_true")\n'
                "    return parser\n\n\n"
                "def main(parser, holder):\n"
                "    args = parser.parse_args()\n"
                "    if args.n5_shadow:\n"
                "        produce = holder\n"
                "        consume(produce)\n"
                "    return 0\n\n\n"
                "def consume(holder):\n    return holder.motor_context\n"
            ),
        },
    )
    sites = [{"file": "taiji/prod.py", "line": 6, "function": "consume"}]
    rows = module.classify(gated_root, sites)
    assert rows[0]["verdict"] == "flag_gated", rows[0]
    assert "n5_shadow" in rows[0]["gating_flags"], rows[0]

    ungated_root = _fixture(
        tmp_path / "ungated",
        {
            "taiji/prod.py": _PRODUCT,
            "scripts/training/train_seed_corpus.py": (
                "import argparse\n\n\n"
                "def build_parser():\n"
                "    parser = argparse.ArgumentParser()\n"
                '    parser.add_argument("--n5-shadow", action="store_true")\n'
                "    return parser\n\n\n"
                "def main(parser, holder):\n"
                "    args = parser.parse_args()\n"
                "    consume(holder)\n"
                "    return 0\n\n\n"
                "def consume(holder):\n    return holder.motor_context\n"
            ),
        },
    )
    flipped = module.classify(ungated_root, sites)
    assert flipped[0]["verdict"] == "reachable_in_default_training_run", flipped[0]
    assert flipped[0]["gated_callers"] == 0, flipped[0]


def test_test_and_instrument_callers_never_light_up_default_face(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": _PRODUCT,
            "tests/taiji_native/t_prod.py": (
                "def test_consume(holder):\n    assert consume(holder) == 1\n"
            ),
            "scripts/training/eval_something.py": "def probe(holder):\n    return consume(holder)\n",
        },
    )
    rows = module.classify(root, [{"file": "taiji/prod.py", "line": 6, "function": "consume"}])
    assert rows[0]["verdict"] == "no_default_face_caller", rows[0]
    assert rows[0]["default_face_callers"] == 0, rows[0]
    assert sum(rows[0]["caller_faces"].values()) == 2, rows[0]


def test_call_inside_a_condition_is_not_itself_a_gate(tmp_path: Path) -> None:
    """`if consume(holder):` 里的调用者**没有**被旗标包住 ⇒ 仍算默认可达。"""

    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": _PRODUCT,
            "taiji/caller.py": (
                "def run(holder):\n"
                "    if consume(holder):\n"
                "        return 1\n"
                "    return 0\n"
            ),
        },
    )
    rows = module.classify(root, [{"file": "taiji/prod.py", "line": 6, "function": "consume"}])
    assert rows[0]["verdict"] == "reachable_at_default", rows[0]
    assert rows[0]["gating_flags"] == [], rows[0]


def test_gate_detection_requires_a_real_default_off_flag(tmp_path: Path) -> None:
    """门控旗标只能来自同文件 argparse 的 store_true／default=False；没有旗标的文件不许判 flag_gated。"""

    import ast

    module = _load()
    with_flag = ast.parse(
        'parser.add_argument("--x", action="store_true")\n'
        'parser.add_argument("--y", default=False)\n'
        'parser.add_argument("--z", default="on")\n'
    )
    assert module._flag_dests(with_flag) == {"x", "y"}, module._flag_dests(with_flag)
    no_flag = ast.parse('parser.add_argument("--z", default="on")\n')
    assert module._flag_dests(no_flag) == set()

    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": _PRODUCT,
            "scripts/training/train_seed_corpus.py": (
                "def run(holder):\n"
                "    flag = True\n"
                "    if flag:\n"
                "        consume(holder)\n"
                "    return 0\n"
            ),
        },
    )
    rows = module.classify(root, [{"file": "taiji/prod.py", "line": 6, "function": "consume"}])
    assert rows[0]["verdict"] == "reachable_in_default_training_run", rows[0]


def test_sites_without_any_caller_get_their_own_exit(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(tmp_path, {"taiji/prod.py": _PRODUCT})
    rows = module.classify(root, [{"file": "taiji/prod.py", "line": 6, "function": "consume"}])
    assert rows[0]["verdict"] == "unreachable_in_tracked_code", rows[0]
    assert rows[0]["caller_count"] == 0, rows[0]


def test_verdict_vocabulary_is_the_only_one_allowed(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": _PRODUCT,
            "taiji/caller.py": "def run(holder):\n    return consume(holder)\n",
        },
    )
    rows = module.classify(root, [{"file": "taiji/prod.py", "line": 6, "function": "consume"}])
    assert all(row["verdict"] in module.VERDICTS for row in rows)
    sites = [{"file": "taiji/prod.py", "line": 6, "function": "consume"}]
    report = module.build_report(root, {"motor_context": sites})
    assert report["tally"]["reachable_at_default"] == 1, report["tally"]
    assert report["no_caller_sites"] == [] and report["indeterminate_sites"] == []
    assert len(report["limits"]) >= 3, report["limits"]


def test_real_repository_face_reproduces_the_published_reading() -> None:
    """真面必须复现 ㊵-699 那条已发表读数：`propose_adaptive_residual_growth_candidate` 判 flag_gated。"""

    module = _load()
    assert WRITER_REPORT.is_file(), "输入件不在仓里：先跑 audit_taiji_write_only_key_face"
    face = json.loads(WRITER_REPORT.read_text(encoding="utf-8"))
    sites_by_key = {}
    for row in face["rows"]:
        parsed = []
        for site in row["action_read_sites"]:
            location, _, function = site.partition("@")
            file_name, _, line = location.partition(":")
            parsed.append({"file": file_name, "line": int(line), "function": function})
        sites_by_key[str(row["key"])] = parsed
    report = module.build_report(PROJECT_ROOT, sites_by_key)
    total = sum(report["tally"].values())
    assert total == sum(len(v) for v in sites_by_key.values()), (total, sites_by_key.keys())
    flagged = [
        (key, row)
        for key, rows in report["per_key"].items()
        for row in rows
        if row["verdict"] == "flag_gated"
    ]
    assert len(flagged) == 1, [(key, row["site"]) for key, row in flagged]
    key, row = flagged[0]
    assert key == "last_decision", key
    assert row["site"].endswith("@propose_adaptive_residual_growth_candidate"), row["site"]
    assert row["gating_flags"] == ["n5_shadow"], row["gating_flags"]
    assert row["caller_faces"].get("tests", 0) >= 1, row["caller_faces"]
    assert row["product_ungated_callers"] == 0, row
