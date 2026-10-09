"""G-N2c-2 的落新名能力：`run_taiji_n2b_powerup.py --run-tag`（㊵-564 起跑前的前置仪器改动）。

为什么要钉它：owner 第九次弹窗批的是"**跑一次**"N2 乙档（同一次通电里出巩固前后两态），
而这台驱动今天把四枚落点写死成 `20261008` 的名字 ⇒ 直接起跑会**覆写 run-2 的证据**
（`G-N2c-2` 的机器侧拒绝针对的是 `--checkpoint`，管不到这种模块常量）。所以先加一档开关，
再按它起跑。三条都能为假：

* 不给 tag ⇒ 五个名字与今天逐字相同（旧命令、在盘件不受影响）；
* 给 tag ⇒ 五个**全部**换名，且目录不漂（档进 `checkpoints/`、件进 `reports/`）——
  只换一半会造出"母档与候选档不同源"这种最难查的错，所以断言是**五条一起**；
* 旗标必须从 CLI 走得通（`--help` 里有点名它），并且那枚 guard2 的落盘**不再写死字面路径**。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "training" / "run_taiji_n2b_powerup.py"

PATHS = ("MOTHER_PATH", "CANDIDATE_PATH", "ROLLBACK_PATH", "POWERUP2_REPORT", "GUARD2_REPORT")


def _module():
    spec = importlib.util.spec_from_file_location("n2b_powerup_under_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(module) -> dict[str, Path]:
    return {name: getattr(module, name) for name in PATHS}


def test_empty_tag_leaves_today_names_untouched() -> None:
    module = _module()
    before = _snapshot(module)
    module._apply_run_tag("")
    assert _snapshot(module) == before, "默认支必须逐字等于今天的名字"
    assert all(
        path.name.endswith("20261008.pt") or path.name.endswith("20261008.json")
        for path in before.values()
    ), before


def test_tag_renames_all_five_and_keeps_each_in_its_directory() -> None:
    module = _module()
    before = _snapshot(module)
    module._apply_run_tag("n2c1")
    after = _snapshot(module)
    for name in PATHS:
        old, new = before[name], after[name]
        assert new.parent == old.parent, (name, old, new)
        assert new.name == f"{old.stem}_n2c1{old.suffix}", (name, old, new)
    #: 档仍在 checkpoints/、件仍在 reports/——换名不许把落点漂走。
    assert all(
        p.parent.name == "checkpoints" for k, p in after.items() if k.endswith("_PATH")
    ), after
    assert all(p.parent.name == "reports" for k, p in after.items() if k.endswith("_REPORT")), after


def test_flag_is_reachable_from_the_cli() -> None:
    proc = subprocess.run(  # noqa: S603 - 仓内固定脚本，参数不含用户输入
        [sys.executable, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "--run-tag" in proc.stdout, proc.stdout[-400:]


def test_guard_report_no_longer_hardcodes_its_literal_at_the_call_site() -> None:
    """落点常量必须**只住一处**：调用点再出现字面路径，就意味着 `--run-tag` 换不到它。"""

    source = RUNNER.read_text(encoding="utf-8")
    call_sites = [line for line in source.splitlines() if "_write_report(REPORT_DIR /" in line]
    assert not call_sites, call_sites
    assert "    _write_report(GUARD2_REPORT, payload)" in source
