"""`audit_taiji_write_only_key_face.py` 的契约册（G-N1c-2 的长驻执行体）。

这册测要钉住四件事，每一件都对应一次我自己犯过的错：

① **赋值目标不是读者**——落地前仪器把 `self._last_decision = decision` 数成第 3 层读者，
   于是「产品在写、没人读」的键被仪器自己刷成「已消费」（比漏报更贵，因为它让
   ``write_only_in_product`` 永远读不出来）。
② **读者分三层**：生产者／薄访问器（第 2 层，只是原样转返回）／读了以后带动作（第 3 层）。
   把第 2 层当消费者会误杀真有读者的键，把第 3 层漏掉会误判「没人消费」。
③ **判定权只在产品面**：测试与训练脚本里的读取不算产品消费。
④ **默认位可达性不许猜**：仪器只把第 3 层点位出版出来，要求逐枚登记；没登记就如实标未登记，
   既不读成「可达」也不读成「不可达」（㊵-698／㊵-699 那条越界死讯的成因就是把这两句并成一句）。

夹具全部走 `taiji/`＋`seed/`＋`tests/` 三张面，且都要 `git add`（扫描面按版本控制取）。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_write_only_key_face.py"


def _load():
    spec = importlib.util.spec_from_file_location("n1_04_write_only_key_face", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True)


def _fixture(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", str(tmp_path)], capture_output=True, check=True)
    _git(tmp_path, "add", ".")
    return tmp_path


def _row(report_rows, key: str) -> dict:
    hits = [row for row in report_rows if row["key"] == key]
    assert len(hits) == 1, (key, [row["key"] for row in report_rows])
    return hits[0]


def test_assignment_target_is_not_counted_as_a_reader(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": (
                "class Holder:\n"
                "    def __init__(self):\n"
                "        self.motor_context = None\n"
                "    def observe(self, value):\n"
                "        self.motor_context = value\n"
                "        return None\n"
            )
        },
    )
    rows = module.judge(module.scan_keys(root, ("motor_context",)), {})
    row = _row(rows, "motor_context")
    assert row["product_writer_sites"] == 2, row
    assert row["product_field_layer3_action_reads"] == 0, row
    assert row["product_verdict"] == "write_only_in_product", row


def test_thin_accessor_is_layer_two_not_a_consumer(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": (
                "class Trigger:\n"
                "    @property\n"
                "    def motor_context(self):\n"
                "        return self._motor_context\n\n\n"
                "class Model:\n"
                "    @property\n"
                "    def shown(self):\n"
                "        return self.trigger.motor_context\n"
            )
        },
    )
    rows = module.judge(module.scan_keys(root, ("motor_context",)), {})
    row = _row(rows, "motor_context")
    assert row["product_field_layer3_action_reads"] == 0, row
    assert row["product_field_layer2_alias_reads"] == 2, row
    assert row["product_verdict"] == "alias_only_in_product", row


def test_value_that_flows_into_a_branch_is_layer_three(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": (
                "def act(model):\n"
                "    decision = model.motor_context\n"
                "    if decision is None:\n"
                "        raise RuntimeError('missing')\n"
                "    return decision\n"
            )
        },
    )
    rows = module.judge(module.scan_keys(root, ("motor_context",)), {})
    row = _row(rows, "motor_context")
    assert row["product_field_layer3_action_reads"] == 1, row
    assert row["product_verdict"] == "consumed_in_product", row
    assert row["reachability_missing_for"], "第三层点位必须要求登记可达性"


def test_local_name_read_alone_does_not_light_up_field_consumption(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": (
                "def build(confidence, raw):\n"
                "    expected_reward = confidence * raw\n"
                "    return make_record(expected_reward=float(expected_reward))\n"
            )
        },
    )
    rows = module.judge(module.scan_keys(root, ("expected_reward",)), {})
    row = _row(rows, "expected_reward")
    assert row["product_field_layer3_action_reads"] == 0, row
    assert row["product_local_name_reads"] >= 1, row
    assert row["product_verdict"] == "produced_and_used_locally_only", row


def test_reads_only_in_tests_do_not_count_as_product_consumption(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": (
                "def fill(state, value):\n" "    state.time_code = value\n" "    return state\n"
            ),
            "tests/t_spec.py": (
                "def test_time_code(fill):\n"
                "    state = fill(object(), 7)\n"
                "    assert state.time_code == 7\n"
            ),
        },
    )
    rows = module.judge(module.scan_keys(root, ("time_code",)), {})
    row = _row(rows, "time_code")
    assert row["product_field_layer3_action_reads"] == 0, row
    assert row["nonproduct_field_layer3_action_reads"] == 1, row
    assert row["product_verdict"] == "write_only_in_product", row


def test_zero_hits_fail_closed_instead_of_reading_as_no_write_only_keys(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(tmp_path, {"taiji/quiet.py": "UNRELATED = 1\n"})
    try:
        module.build_report(root, module.NAMED_KEYS, {})
    except ValueError as exc:
        assert "一枚都没找到" in str(exc), exc
    else:
        raise AssertionError("零命中必须响亮失败，不许落半件")


def test_reachability_ledger_moves_both_branches(tmp_path: Path) -> None:
    module = _load()
    root = _fixture(
        tmp_path,
        {
            "taiji/prod.py": (
                "def act(model):\n"
                "    decision = model.motor_context\n"
                "    if decision:\n"
                "        return 1\n"
                "    return 0\n"
            )
        },
    )
    face = module.scan_keys(root, ("motor_context",))
    unregistered = _row(module.judge(face, {}), "motor_context")
    assert len(unregistered["reachability_missing_for"]) == 1, unregistered
    assert unregistered["reachability_registered"] == 0, unregistered

    site = unregistered["reachability_missing_for"][0]
    ledger = {"motor_context": {f"motor_context@{site}": "默认装配可达，出处 taiji/prod.py:act"}}
    registered = _row(module.judge(face, ledger), "motor_context")
    assert registered["reachability_missing_for"] == [], registered
    assert registered["reachability_registered"] == 1, registered
    assert f"motor_context@{site}" in registered["reachability"], registered


def test_named_keys_are_pinned_to_the_file_that_declares_them() -> None:
    """每一枚键的名字都要能在它**声明处**现读到。

    我早先把这张表写成「PLAN-N1-03 §1／§2 点名」——现读否证（那件里只出现 5 枚，其余 8 枚
    在整棵 plans 树里零命中）。所以钉的是产品码里的声明，不是文档措辞（㊵-699／㊵-701）。
    """

    module = _load()
    assert len(module.NAMED_KEYS) == 13, module.NAMED_KEYS
    assert set(module.KEY_ORIGINS) == set(module.NAMED_KEYS), "键表与出处表必须同源"
    for key, origin in module.KEY_ORIGINS.items():
        path = PROJECT_ROOT / origin
        assert path.is_file(), (key, origin)
        assert key in path.read_text(encoding="utf-8"), f"{key} 已不在其声明处 {origin}"


def test_alias_extra_name_entry_still_resolves() -> None:
    """`EXTRA_NAMES` 折进来的那条别名访问器必须还在原处（改名或删掉即红，逼我重登记）。"""

    module = _load()
    import ast

    model_source = (PROJECT_ROOT / "taiji" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(model_source)
    for key, extra_names in module.EXTRA_NAMES.items():
        for alias in extra_names:
            bodies = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == alias
            ]
            assert bodies, f"别名访问器 {alias} 已不在 taiji/model.py：登记过期，须重登或撤折叠"
            forwarded = [
                node.attr
                for node in ast.walk(bodies[0])
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
            ]
            assert key in forwarded or f"_{key}" in forwarded, (alias, forwarded)

    # 判别力自证：把那条属性改名后，同一谓词必须查不到（否则"别名仍在案"是恒真式）。
    mutated = model_source.replace(
        f"def {next(iter(module.EXTRA_NAMES['last_decision']))}", "def renamed_away"
    )
    mutated_tree = ast.parse(mutated)
    assert not [
        node
        for node in ast.walk(mutated_tree)
        if isinstance(node, ast.FunctionDef) and node.name in module.EXTRA_NAMES["last_decision"]
    ], "改名后仍被查到＝这条钉是恒真式"


def test_cli_writes_a_lf_only_artifact(tmp_path: Path) -> None:
    """落件必须是 LF：本仓 `reports/*.json` 钉了 `text eol=lf`，用 `write_text` 不带
    `newline` 在 Windows 上会翻成 CRLF，于是每次重跑都留下一枚幻影脏文件（DEBT-G35／G50 那一族）。"""

    target = tmp_path / "face.json"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-report", str(target), "--overwrite"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    blob = target.read_bytes()
    assert blob.count(b"\r") == 0, blob.count(b"\r")
    assert json.loads(blob.decode("utf-8"))["files_parsed"] > 1000


def test_cli_refuses_to_overwrite_a_sealed_report(tmp_path: Path) -> None:
    target = tmp_path / "sealed.json"
    target.write_text(json.dumps({"placeholder": True}), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--out-report", str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=PROJECT_ROOT,
    )
    assert proc.stdout is not None, proc.stderr
    assert proc.returncode == 2, (proc.returncode, proc.stderr[-400:])
    assert "refusing" in proc.stderr, proc.stderr[-400:]
    assert json.loads(target.read_text(encoding="utf-8")) == {"placeholder": True}


def test_real_repository_face_is_complete_and_verdicts_are_pinned() -> None:
    module = _load()
    report = module.build_report(PROJECT_ROOT, module.NAMED_KEYS, {})
    assert report["files_parsed"] > 1000, report["files_parsed"]
    rows = report["rows"]
    assert len(rows) == 13, [row["key"] for row in rows]
    assert all(row["product_verdict"] in module.PRODUCT_VERDICTS for row in rows)
    assert all(row["verdict"] in module.VERDICTS for row in rows)
    assert sum(report["product_tally"].values()) == 13, report["product_tally"]

    expected = _row(rows, "expected_reward")
    assert expected["product_verdict"] == "produced_and_used_locally_only", expected
    for key in ("answer_source", "unique_documents"):
        assert _row(rows, key)["product_verdict"] == "read_only_outside_product", key

    last = _row(rows, "last_decision")
    assert last["product_verdict"] == "consumed_in_product", last
    assert last["product_field_layer2_alias_reads"] >= 1, "薄访问器那一层必须被单独数到"
    assert any(
        site.endswith("@propose_adaptive_residual_growth_candidate")
        for site in last["action_read_sites"]
    ), last["action_read_sites"]
    assert "taiji/model.py" in {site.split(":")[0] for site in last["action_read_sites"]}
    assert _row(rows, "motor_context")["product_verdict"] == "consumed_in_product"
    assert len(report["limits"]) >= 3, report["limits"]
