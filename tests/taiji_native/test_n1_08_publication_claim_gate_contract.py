"""`audit_taiji_n1_03_publication_claim_gate.py` 的契约册：三态不许折叠、旧件不许拿来判今天、空面不许默认全仓。

这册要钉住三件会伪装成"门很严"的事：
①把 `not_established`（没有可读物）读成红或绿——读成绿就是 ㊵-708 那条假绿，读成红会让人以为前置被否证；
②拿一份**旧读数**判今天的前置（仪器已升版）——所以 `format` 与生产它的仪器不同源时必须响亮失败；
③扫描面一旦默认全仓，就会命中**预注册自己抄了一遍的那几句宣称**（现读 2／1／2 处），门永远红在身上。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "audit_taiji_n1_03_publication_claim_gate.py"
STUB_FORMAT = "stub-producer-format-v1"
#: 今天（2026-10-11）真面的态。它**只用于**说明"这条常驻断言当前看到什么"，
#: 断言本身不收死这个值——③ 拿到第一枚载体后这里会变，而那不该算红。
VERDICT_NOW = {"pending_not_claimed"}


def _load():
    spec = importlib.util.spec_from_file_location("n1_08_publication_claim_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _artifact(module: Path, name: str, payload: dict) -> None:
    path = module / "reports" / Path(name).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8", newline="\n")


def _root(tmp_path: Path, role: str, provenance: str, rows: list[dict]) -> Path:
    """造一个三面齐全、且 `format` 与**桩件生产者**同源的根。"""

    module = _load()
    for rel in module.PRODUCER_MODULES.values():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f'REPORT_FORMAT = "{STUB_FORMAT}"\n', encoding="utf-8", newline="\n")
    _artifact(
        tmp_path,
        module.ROLE_FACE,
        {"format": STUB_FORMAT, "j_n1c_2": {"verdict": role}},
    )
    _artifact(tmp_path, module.WRITE_ONLY_FACE, {"format": STUB_FORMAT, "rows": rows})
    _artifact(
        tmp_path,
        module.PROVENANCE_FACE,
        {"format": STUB_FORMAT, "carrier_face": {"verdict": provenance, "carrier_count": 0}},
    )
    doc = tmp_path / "state.md"
    doc.write_text("# 状态面\n\n今天什么也没宣称。\n", encoding="utf-8", newline="\n")
    return tmp_path


GOOD_ROWS = [
    {
        "key": "answer_source",
        "writer_sites": 3,
        "layer3_action_reads": 2,
        "layer2_alias_reads": 0,
        "unclassified_reads": 1,
    },
]


def test_missing_artifact_is_loud_not_silent(tmp_path: Path) -> None:
    module = _load()
    try:
        module._read_artifact(tmp_path, module.ROLE_FACE)
    except ValueError as exc:
        assert "前置件缺失" in str(exc), exc
    else:
        raise AssertionError("缺件必须响亮失败——不许把「没读数」读成「没宣称就没问题」")


def test_stale_format_is_refused(tmp_path: Path) -> None:
    """仪器升了版而件还是旧的 ⇒ 不许拿旧件判今天的前置。"""

    module = _load()
    root = _root(tmp_path, "table_registered", "consistent_within_carriers", GOOD_ROWS)
    path = root / module.ROLE_FACE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["format"] = "some-older-format"
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")
    try:
        module._read_artifact(root, module.ROLE_FACE)
    except ValueError as exc:
        assert "不同源" in str(exc), exc
    else:
        raise AssertionError("format 与生产者不同源必须拒绝")


def test_off_dictionary_verdict_raises(tmp_path: Path) -> None:
    module = _load()
    try:
        module.judge_role({"j_n1c_2": {"verdict": "table_perhaps_registered"}})
    except ValueError as exc:
        assert "跑出词典" in str(exc), exc
    else:
        raise AssertionError("判级词跑到词典外必须拒绝，不许默认算 satisfied")


def test_no_carrier_is_neither_red_nor_green(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path, "table_registered", "no_carrier_outside_weight_artifacts", GOOD_ROWS)
    report = module.adjudicate(root, ["state.md"])
    states = {member["member"]: member["state"] for member in report["members"]}
    third = [value for key, value in states.items() if key.startswith("§7③")][0]
    assert third == "not_established", states
    assert report["verdict"] == "pending_not_claimed", report
    assert report["publishable"] is False, report
    assert report["state_counts"] == {"satisfied": 2, "not_established": 1, "violated": 0}, report


def test_all_satisfied_and_claimed_is_publishable(tmp_path: Path) -> None:
    module = _load()
    root = _root(tmp_path, "table_registered", "consistent_within_carriers", GOOD_ROWS)
    (root / "state.md").write_text(
        "# 状态面\n\nV-11 与 V-19 转为已展开且有承载表。\n", encoding="utf-8", newline="\n"
    )
    report = module.adjudicate(root, ["state.md"])
    assert report["verdict"] == "publishable_and_claimed", report
    assert report["claims"]["claim_hits"], report


def test_claim_without_precondition_is_the_red_branch(tmp_path: Path) -> None:
    """门能为假的主支：前置未齐却已经把它说出去 ⇒ rc=1。"""

    module = _load()
    root = _root(tmp_path, "table_registered", "no_carrier_outside_weight_artifacts", GOOD_ROWS)
    (root / "state.md").write_text(
        "# 状态面\n\nN1 已把回答链与角色通路做完。\n", encoding="utf-8", newline="\n"
    )
    report = module.adjudicate(root, ["state.md"])
    assert report["verdict"] == "claims_without_precondition", report
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--doc",
            "state.md",
            "--out-report",
            "out.json",
        ],
        cwd=root,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 1, run.stderr[:200]


def test_negated_quotation_is_skipped_and_counted(tmp_path: Path) -> None:
    """预注册里那种"不宣布…"的引用不许被读成宣称——但豁免数量必须出版。"""

    module = _load()
    root = _root(tmp_path, "table_registered", "no_carrier_outside_weight_artifacts", GOOD_ROWS)
    (root / "state.md").write_text(
        "# 状态面\n\n不宣布 V-11 与 V-19 转为已展开且有承载表。\n", encoding="utf-8", newline="\n"
    )
    report = module.adjudicate(root, ["state.md"])
    assert report["claims"]["claim_hits"] == [], report["claims"]
    assert report["claims"]["lines_skipped_as_negated"] == 1, report["claims"]
    assert report["verdict"] == "pending_not_claimed", report


def test_empty_scan_face_raises(tmp_path: Path) -> None:
    module = _load()
    try:
        module.scan_claims(tmp_path, [])
    except ValueError as exc:
        assert "扫描面为空" in str(exc), exc
    else:
        raise AssertionError("空面必须拒绝——默认全仓会命中预注册自己抄的那几句")


def test_write_only_row_without_either_side_is_violated() -> None:
    module = _load()
    rows = [{"key": "last_decision", "writer_sites": 0, "layer3_action_reads": 4}]
    judged = module.judge_write_only({"rows": rows})
    assert judged["state"] == "violated", judged

    both = module.judge_write_only(
        {"rows": [{"key": "k", "writer_sites": 2, "layer3_action_reads": 1}]}
    )
    assert both["state"] == "satisfied", both


def test_empty_key_face_is_not_established_not_satisfied() -> None:
    module = _load()
    judged = module.judge_write_only({"rows": []})
    assert judged["state"] == "not_established", judged


def test_real_face_reproduces_today_reading() -> None:
    """真面锚点：三条前置＝2 satisfied＋1 not_established，三件状态面文档零宣称。"""

    module = _load()
    docs = [
        "plans/active/roadmap/03_CURRENT_EXECUTION.md",
        "plans/active/roadmap/09_NEXT_MAINLINE_PLAN.md",
        "plans/reference/N_MAINLINE_COMPLETION_AUDIT_20261011.md",
    ]
    report = module.adjudicate(PROJECT_ROOT, docs)
    #: 这条常驻断言**不许**在进展到来时变红：今天真面是 `pending_not_claimed`，
    #: 等 ③ 拿到第一枚载体就会变成 `publishable_not_claimed`／`publishable_and_claimed`。
    #: 真正不许发生的是"前置未齐却已宣称"，以及"零命中被读成通过"——所以钉的是这两条。
    assert report["verdict"] != "claims_without_precondition", report
    assert report["verdict"] in VERDICT_NOW | {
        "publishable_not_claimed",
        "publishable_and_claimed",
    }, report["verdict"]
    assert report["publishable"] or not report["claims"]["claim_hits"], report
    assert set(report["state_counts"]) == {"satisfied", "not_established", "violated"}, report
    assert sum(report["state_counts"].values()) == 3, report
    assert report["state_counts"]["violated"] == 0, report
    assert report["claims"]["docs_scanned"] == 3, report["claims"]
    assert report["claims"]["lines_skipped_as_negated"] == 0, report["claims"]
    #: 精确不变量而不是猜一个下限：扫到的行数必须**等于**那三件各自的行数之和。
    #: （03 那份是"一条 note＝一个超长物理行"的形状，64 行而已——先按行数猜下限就会像刚才那样错。）
    expected_lines = sum(
        len((PROJECT_ROOT / doc).read_text(encoding="utf-8").splitlines()) for doc in docs
    )
    assert report["claims"]["lines_scanned"] == expected_lines, (
        report["claims"],
        expected_lines,
    )
    assert expected_lines > 100, expected_lines
    assert len(report["what_this_does_not_prove"]) >= 3, report
    # 预注册件里那几句是**规范原文**，若把它扫进来门就会自指命中——这里钉住"面必须显式给"。
    prereg = (PROJECT_ROOT / report["prereg"]).read_text(encoding="utf-8")
    assert any(claim in prereg for claim in module.CLAIMS), "预注册本就该带着那几句原文"


def test_cli_refuses_to_overwrite_and_writes_lf(tmp_path: Path) -> None:
    out = tmp_path / "gate.json"
    out.write_text("{}", encoding="utf-8", newline="\n")
    docs = ["plans/active/roadmap/03_CURRENT_EXECUTION.md"]
    refused = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--doc",
            docs[0],
            "--out-report",
            str(out),
        ],
        capture_output=True,
        check=False,
    )
    assert refused.returncode == 2, refused.stderr[:200]
    written = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--doc",
            docs[0],
            "--out-report",
            str(out),
            "--overwrite",
        ],
        capture_output=True,
        check=False,
    )
    assert written.returncode == 0, written.stderr[:300]
    assert b"\r\n" not in out.read_bytes()
