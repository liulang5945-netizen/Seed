"""DEBT-I7 (second instance): the canary residue sweep must delete *exactly* its own files.

The session fixture in ``tests/conftest.py`` calls :func:`sweep` at teardown against
``output/manual-r5-canary/``.  A cleanup that deletes too much is worse than no cleanup -- the same
directory holds ``README.md``, ``native-canary.pt``, and another session's in-flight files -- so
both directions are asserted here, against a scratch directory rather than the real one.
"""

from __future__ import annotations

from pathlib import Path

from _canary_sweep import sweep

MY_PID = 4242
OTHER_PID = 9999


def _build(directory: Path) -> dict[str, Path]:
    made: dict[str, Path] = {}
    for name in (
        f"s45-active-{MY_PID}.pt",
        f"s41-runtime-{MY_PID}.pt",
        f"s45-active-{OTHER_PID}.pt",
        "README.md",
        "native-canary.pt",
        f"s45-active-{MY_PID}.bak",
        "s45-active-no-pid.pt",
    ):
        made[name] = directory / name
        made[name].write_text("x", encoding="utf-8")
    for name in (f"s41-store-{MY_PID}", f"s41-store-{OTHER_PID}"):
        store = directory / name
        store.mkdir()
        (store / "part.pt").write_text("x", encoding="utf-8")
        made[name] = store
    return made


def test_sweep_removes_this_pids_residue_and_nothing_else(tmp_path) -> None:
    made = _build(tmp_path)

    removed = sweep(tmp_path, MY_PID)

    assert removed == [
        f"s41-runtime-{MY_PID}.pt",
        f"s41-store-{MY_PID}",
        f"s45-active-{MY_PID}.pt",
    ]
    # 目录是整棵删掉的，不是只删了名字。
    assert not (made[f"s41-store-{MY_PID}"] / "part.pt").exists()
    survivors = sorted(entry.name for entry in tmp_path.iterdir())
    assert survivors == [
        "README.md",
        "native-canary.pt",
        f"s41-store-{OTHER_PID}",
        f"s45-active-{MY_PID}.bak",
        f"s45-active-{OTHER_PID}.pt",
        "s45-active-no-pid.pt",
    ]
    assert (made[f"s41-store-{OTHER_PID}"] / "part.pt").read_text(encoding="utf-8") == "x"


def test_sweep_on_an_absent_directory_is_a_no_op(tmp_path) -> None:
    """目录还不存在（第一次跑、或被别的会话清过）时不许抛异常。"""

    assert sweep(tmp_path / "never-created", MY_PID) == []
