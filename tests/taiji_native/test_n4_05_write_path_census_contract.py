"""N4 写入链普查仪（`audit_taiji_n4_write_path_census.py`）的契约测。

这台仪器只回答"代码里有没有这处写、默认位是什么"，所以**两侧都能为假**是它唯一的资格：

* 名字以 `episodic_` 开头的调用点必须被数到——㊵-643 第一版的正则要求 "episodic" 前面还有字符，
  于是把训练器里真实存在的 `episodic_store.write(...)` 读成 0（**假阴性**，方向是"把有读成没有"）；
* docstring 里提到的同名串**不许**被数到——同一版的正则把 `make_taiji_n4_product_tier_faces.py`
  第 5 行的说明文字当成调用点（**假阳性**）；
* 第三条判定 `write_happens_at_product_default` 必须随默认位翻转（两支都走）；
* 目标文件不在场 ⇒ rc=2 且出件 `census_failed`，那是"没算"，不许读成"没有写入点"。

夹具全部落在 pytest 的 `tmp_path`，不碰 `reports/` 与 `output/`。真树那一支钉的是**现读**计数，
将来挂钩子/拆挂载点会当场红并要求重钉（这是本册的存在理由，不是装饰）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "training" / "audit_taiji_n4_write_path_census.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("n4_02_write_census", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = _load()

CONFIG_TEMPLATE = """
class TaijiConfig:
    episodic_memory_default_mount: bool = {mount}
    episodic_memory_capacity: int = 1024
"""


def _tree(tmp_path: Path, *, mount: bool, body: str) -> Path:
    taiji = tmp_path / "taiji"
    taiji.mkdir(parents=True, exist_ok=True)
    (taiji / "config.py").write_text(
        CONFIG_TEMPLATE.format(mount="True" if mount else "False"), encoding="utf-8", newline="\n"
    )
    (taiji / "adapter.py").write_text(body, encoding="utf-8", newline="\n")
    return tmp_path


CALL_BODY = """
class Adapter:
    def settle(self):
        if self.store is not None:
            episodic_store.write(object())
"""

DOCSTRING_BODY = '''
class Adapter:
    """说明文字里提到 episodic_store.write(object()) 这一行不是调用点。"""

    def settle(self):
        return None
'''


def test_a_name_starting_with_episodic_is_counted_not_missed(tmp_path: Path) -> None:
    payload = MODULE.audit(_tree(tmp_path, mount=False, body=CALL_BODY))
    assert payload["write_site_count"] == 1, payload["write_sites"]
    assert payload["product_write_site_count"] == 1
    assert payload["write_sites"][0]["target"] == "episodic_store"
    assert payload["write_sites"][0]["guarded_by"] == "if self.store is not None:"


def test_a_docstring_mention_is_not_a_call_site(tmp_path: Path) -> None:
    payload = MODULE.audit(_tree(tmp_path, mount=False, body=DOCSTRING_BODY))
    assert payload["write_site_count"] == 0, payload["write_sites"]
    assert payload["verdicts"]["product_chain_has_write_call"] is False


def test_the_default_flip_is_what_changes_the_third_verdict(tmp_path: Path) -> None:
    #: 同一份调用点，只翻默认位 ⇒ 第三条判定必须跟着翻（否则它是恒真式）。
    off = MODULE.audit(_tree(tmp_path / "off", mount=False, body=CALL_BODY))
    on = MODULE.audit(_tree(tmp_path / "on", mount=True, body=CALL_BODY))
    assert off["verdicts"]["write_happens_at_product_default"] is False
    assert on["verdicts"]["write_happens_at_product_default"] is True
    assert on["defaults"]["episodic_memory_default_mount"] is True


def test_the_real_tree_pins_are_read_not_assumed() -> None:
    payload = MODULE.audit(REPO)
    assert payload["defaults"] == {
        "episodic_memory_default_mount": False,
        "episodic_memory_capacity": 1024,
    }, payload["defaults"]
    #: 现读四枚计数：产品链 1 处（`taiji/adapter.py:11699`）、训练链 2 处（:477／:962）、
    #: 挂载点合计 7 枚且 `taiji/adapter.py` 恰 1 枚、扫面 846 枚文件。
    assert payload["write_site_count"] == 4, payload["write_sites"]
    assert payload["product_write_site_count"] == 1
    assert payload["trainer_write_site_count"] == 2
    assert payload["mount_site_count"] == 7
    assert payload["adapter_mount_site_count"] == 1
    assert payload["files_scanned"] > 800
    #: 关键结论：默认位为关 ⇒ "产品链有写的调用点"与"默认下真会写"是两件事，后者为假。
    assert payload["verdicts"] == {
        "product_chain_has_write_call": True,
        "trainer_chain_has_write_call": True,
        "write_happens_at_product_default": False,
    }, payload["verdicts"]
    assert "runtime reachability" in payload["reading_limit"]


def test_a_missing_config_is_census_failed_not_no_write_sites(tmp_path: Path) -> None:
    report = tmp_path / "out.json"
    rc = MODULE.main(["--root", str(tmp_path), "--out-report", str(report)])
    assert rc == 2, rc
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "census_failed"
    assert "write_site_count" not in payload


def test_help_runs_and_lists_both_flags(monkeypatch) -> None:
    captured: list[str] = []
    monkeypatch.setattr(sys.stdout, "write", lambda text: captured.append(text) or len(text))
    try:
        MODULE.main(["--help"])
    except SystemExit as exit_info:
        assert exit_info.code == 0, exit_info.code
    else:
        raise AssertionError("--help 没有按惯例退出 0")
    joined = "".join(captured)
    assert "--out-report" in joined and "--root" in joined, joined[:200]
