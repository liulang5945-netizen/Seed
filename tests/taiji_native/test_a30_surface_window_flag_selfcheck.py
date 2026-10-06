"""`--product-window-steps` 的**自证守卫**（2026-10-03，`DEBT-G30` 同族的第二颗牙）。

起因是一条我差点签收的假对照：给装机信封跑「时序门开 K=128」那一臂时，产出的件与门不开的
那件**逐字节相同**（`19,677` 对 `19,677`），全文找不到 `128`——因为控制臂当时不接这个旗标，
而治疗臂列表在不给 `--circuit` 时是空的。若照原样引用，结论就成了"开了时序门表层成句没变化"，
而实际是**门根本没被走到**。这类旗标只能靠仪器自己拒绝落盘来防。

守卫钉三条：不给旗标 ⇒ 放行；给了但没臂带 ⇒ 判假；给了且有臂带 ⇒ 放行。
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSTRUMENT_REL = "scripts/training/score_taiji_r2_copy_surface_extension.py"
INSTRUMENT = PROJECT_ROOT / INSTRUMENT_REL


def _instrument():
    spec = importlib.util.spec_from_file_location("_surface_ext_under_test", INSTRUMENT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scan_face_is_the_tracked_file() -> None:
    tracked = (
        subprocess.run(
            ["git", "ls-files", INSTRUMENT_REL], cwd=PROJECT_ROOT, capture_output=True, check=True
        )
        .stdout.decode()
        .strip()
    )
    assert tracked == INSTRUMENT_REL, f"仪器不在版本控制面上（扫到 {tracked!r}）"


def test_three_branches_of_the_self_check() -> None:
    """三条都要能为 false：恒真或恒假的自证都等于没有自证。"""

    mod = _instrument()
    assert mod._window_flag_honored(None, [{"product_window_steps": None}]) is True
    assert mod._window_flag_honored(128, [{"product_window_steps": None}]) is False
    assert mod._window_flag_honored(128, [{"product_window_steps": 128}]) is True
    #: 治疗臂里带上也算：只有**没有任何一臂**带上才判假
    assert (
        mod._window_flag_honored(
            128, [{"product_window_steps": None}, {"product_window_steps": 128}]
        )
        is True
    )


def test_control_arm_now_receives_the_flag() -> None:
    """成因级钉法：控制臂调用必须把旗标传进去，否则自证会永远走"判假"那支、门再也量不了。"""

    source = INSTRUMENT.read_text(encoding="utf-8")
    control_call = source[source.index("control = run_arm(") :]
    control_call = control_call[: control_call.index(")\n") + 1]
    assert "product_window_steps=args.product_window_steps" in control_call, control_call


def test_the_check_runs_before_any_write(tmp_path: Path) -> None:
    """自证必须发生在落件之前——已经写出假对照再报错就晚了。"""

    source = INSTRUMENT.read_text(encoding="utf-8")
    check_at = source.index("_window_flag_honored(args.product_window_steps")
    write_at = source.index("out.write_text(json.dumps(report")
    assert check_at < write_at, (check_at, write_at)
