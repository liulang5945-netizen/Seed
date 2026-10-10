"""`check_taiji_n6_head_face_parity.py`（J-N6a-4 的尺）的契约测。

这把尺要钉两件事，缺一件就只是"自我感觉良好"：

1. **它复算了一条已入库的钉子**——`position_off` 的摘要必须等于
   `tests/taiji_native/test_n4_05_product_default_mount_contract.py` 里的 `BASELINE_DIGEST`。
   出处从那份文件**现读**（AST 取常量），所以别人改了钉子这里当场红，而不是两处分头过期。
2. **它有动态范围**——同一把尺下 `position_off != position_on`；否则"逐位相同"可以只是"根本没动"。

再加一条本仪器的存在理由：任何一档在两棵树上取不到摘要 ⇒ `rc=2`＋`parity_indeterminate`。
㊵-651 实测过这个形状（发育档第一版两棵树都返回 `None`，`None == None` 会读成真绿）。

**这条"identical"断言在实施格入库之后会不会变成恒真？** 不会失去意义，但作用面要写清：
它此后守的是**尚未提交的工作树改动**——任何人改了 `taiji/`／`seed/` 里影响默认观测形状的东西，
在提交之前就会在这里红（HEAD 与工作树本来就应当相同）。持久有效的两条分辨力是上面那两条：
钉子复现（尺没有换）与位置开关可见（尺有动态范围）。对照用的 commit 由件自述（`head_commit`），
本轮封的那枚是 `2f3f2c54b`——它只动 `taiji-harness/`（并行会话的 UI 修复），不含本件的产品改动，
所以"HEAD 的 `taiji/`＋`seed/` vs 含改动的工作树"正是本条要问的那个对照。

整件对表要起两个面进程（HEAD 解包树＋工作树），所以全套共用一次运行结果（模块级缓存）。
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "check_taiji_n6_head_face_parity.py"
RULER_HOLDER = REPO / "tests" / "taiji_native" / "test_n4_05_product_default_mount_contract.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n5_20_head_face_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _committed_baseline_digest() -> str:
    tree = ast.parse(RULER_HOLDER.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "BASELINE_DIGEST":
                    value = node.value
                    assert isinstance(value, ast.Constant) and isinstance(value.value, str)
                    return str(value.value)
    raise AssertionError(f"{RULER_HOLDER.name} 里找不到 BASELINE_DIGEST ⇒ 尺的出处消失了")


@pytest.fixture(scope="module")
def parity_payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    report = tmp_path_factory.mktemp("n6_parity") / "parity.json"
    rc = MODULE.main(["--repo", str(REPO), "--out-report", str(report)])
    payload = json.loads(report.read_text(encoding="utf-8"))
    #: 出版 rc 与判定同源——测里不许另算一套。
    assert rc == MODULE.parity_rc(payload), (rc, payload["verdicts"])
    return payload


def test_the_ruler_reproduces_the_committed_baseline_digest(parity_payload: dict[str, Any]) -> None:
    expected = _committed_baseline_digest()
    work = parity_payload["worktree_digests"]
    head = parity_payload["head_digests"]
    assert work["position_off"] == expected, (work["position_off"], expected)
    assert head["position_off"] == expected, (head["position_off"], expected)
    assert parity_payload["verdicts"]["position_off"] == "identical", parity_payload["verdicts"]


def test_the_ruler_can_see_the_position_switch(parity_payload: dict[str, Any]) -> None:
    #: 反向证据：这把尺不是"什么都没测"。位置输入开／关必须给出**不同**摘要。
    assert parity_payload["worktree_position_switch_is_visible"] is True, parity_payload[
        "worktree_digests"
    ]


def test_every_selected_mode_is_judged_not_silently_absent(parity_payload: dict[str, Any]) -> None:
    assert set(parity_payload["verdicts"]) == set(parity_payload["modes"]), parity_payload
    assert all(value != "unmeasured" for value in parity_payload["verdicts"].values()), (
        parity_payload["verdicts"],
        parity_payload["face_errors"],
    )
    #: 三档都得有非空摘要；任何一枚 `None` 都意味着挂载序列缺一环（㊵-651 实测形状）。
    for row in (parity_payload["head_digests"], parity_payload["worktree_digests"]):
        assert all(value is not None for value in row.values()), row


def test_an_unmeasured_mode_cannot_read_as_parity() -> None:
    """这一支不需要起进程：判定函数本身必须把 `None == None` 挡在门外。"""

    assert MODULE.parity_rc({"verdicts": {"position_off": "unmeasured"}}) == 2
    assert MODULE.parity_rc({"verdicts": {"position_off": "identical"}}) == 0
    assert MODULE.parity_rc({"verdicts": {"position_off": "diverged"}}) == 1
    assert (
        MODULE.parity_rc({"verdicts": {"position_off": "identical", "position_on": "unmeasured"}})
        == 2
    )


def test_an_unknown_mode_name_is_refused_by_the_cli(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as boom:
        MODULE.main(
            ["--repo", str(REPO), "--out-report", str(tmp_path / "x.json"), "--modes", "bogus"]
        )
    assert boom.value.code == 2, boom.value


def test_help_lists_every_flag() -> None:
    proc = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")[:400]
    text = proc.stdout.decode("utf-8", errors="replace")
    for flag in ("--out-report", "--repo", "--modes"):
        assert flag in text, (flag, text)
