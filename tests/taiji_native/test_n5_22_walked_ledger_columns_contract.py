"""PLAN-N5-12 批次 A 的契约册：`probe_taiji_a27_walked_ledger.py` 的两列分开＋落盘护栏。

要判的是三件事，各自都能为假：
* `absent` 这一桶今天同时装「零件未装配」与「方法名点错」，仪器对点错名的零件**没有判别力**
  （无论装没装都只会报 absent）⇒ 分成 `not_attached`／`method_not_found` 两列，且 `absent` 保留为二者之并；
* homeostatic 那格点的 `step` 在 `HomeostaticController` 上不存在 ⇒ 必须改成真实存在的方法名，
  并且要有一条**用真类**的正对照，防止日后又漂回错名；
* `--out-report` 的默认值指着已封存的 tracked 件，直接重跑会覆写历史读数 ⇒ 默认落点必须不再指向它，
  且「目标已存在」时必须响亮拒绝。
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "training" / "probe_taiji_a27_walked_ledger.py"
SEALED = "reports/taiji_a27_walked_ledger_20260928.json"


def _load_probe():
    spec = importlib.util.spec_from_file_location("n5_12_walked_ledger_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Holder:
    """带一个真方法的零件：正对照用。"""

    def __init__(self) -> None:
        self.calls = 0

    def update(self, *_args, **_kwargs) -> str:
        self.calls += 1
        return "ok"


class _Root:
    def __init__(self, **slots) -> None:
        for name, value in slots.items():
            setattr(self, name, value)


def test_method_present_is_counted_and_in_neither_column() -> None:
    probe = _load_probe()
    holder = _Holder()
    root = _Root(_probe_part=holder)
    counter = {"probe_part": 0}
    installed: list[tuple[object, str]] = []
    probe.PARTS = (("probe_part", ("_probe_part",), "update"),)
    not_attached, method_not_found = probe._install(root, counter, installed)
    try:
        assert not_attached == []
        assert method_not_found == []
        #: 打包装本身不产生计数——计数只在被调时涨（这正是"被走到"的含义）。
        assert counter["probe_part"] == 0
        holder.update()
        assert counter["probe_part"] == 1
        holder.update()
        assert counter["probe_part"] == 2
    finally:
        probe._restore(installed)


def test_holder_missing_lands_in_not_attached_not_method_not_found() -> None:
    probe = _load_probe()
    root = _Root()  # 没有 _probe_part 这个属性
    counter = {"probe_part": 0}
    probe.PARTS = (("probe_part", ("_probe_part",), "update"),)
    not_attached, method_not_found = probe._install(root, counter, [])
    assert not_attached == ["probe_part"]
    assert method_not_found == []


def test_holder_present_but_method_missing_is_the_other_column() -> None:
    probe = _load_probe()
    root = _Root(_probe_part=_Holder())
    counter = {"probe_part": 0}
    probe.PARTS = (("probe_part", ("_probe_part",), "recall"),)
    not_attached, method_not_found = probe._install(root, counter, [])
    assert method_not_found == ["probe_part"]
    assert not_attached == []


def test_absent_key_is_still_the_union_of_both_columns() -> None:
    """必需键不扩、历史列仍可对照：`absent` 必须是两列之并。"""

    probe = _load_probe()
    root = _Root(_present=_Holder())
    counter = {"present": 0, "gone": 0, "badname": 0}
    probe.PARTS = (
        ("present", ("_present",), "update"),
        ("gone", ("_gone",), "update"),
        ("badname", ("_present",), "recall"),
    )
    not_attached, method_not_found = probe._install(root, counter, [])
    report_absent = sorted(set(not_attached) | set(method_not_found))
    assert report_absent == ["badname", "gone"]
    assert not_attached == ["gone"]
    assert method_not_found == ["badname"]


def test_homeostatic_label_names_a_method_the_real_class_has() -> None:
    """用**真类**做正对照：homeostatic 那格必须点一个 `HomeostaticController` 上存在的方法。"""

    from taiji.homeostasis import HomeostaticController, HomeostaticState

    probe = _load_probe()
    labels = {label: (path, method) for label, path, method in probe.PARTS}
    assert "_homeostatic_controller" in labels["homeostatic_controller"][0]
    method = labels["homeostatic_controller"][1]
    controller = HomeostaticController()
    assert callable(
        getattr(controller, method, None)
    ), f"PARTS 把 homeostatic 格点在 {method} 上，而 HomeostaticController 没有这个方法"
    root = _Root(_homeostatic_controller=controller)
    counter = {"homeostatic_controller": 0}
    probe.PARTS = (("homeostatic_controller", ("_homeostatic_controller",), method),)
    not_attached, method_not_found = probe._install(root, counter, [])
    assert (not_attached, method_not_found) == ([], [])
    controller.update(HomeostaticState(tick=0))
    assert counter["homeostatic_controller"] == 1


def test_every_probed_method_exists_on_the_class_that_fills_its_slot() -> None:
    """整表自证：26 枚零件里不许再有「方法名在该类上不存在」的形状。

    取法＝对每格构造一个只带该方法的假 holder 会失去判别力，所以这里按**静态归属**做：
    对能在 `taiji/adapter.py` 里解析出被赋的类的格子，逐个核方法名是否定义在那枚类上。
    解析不出类的格子（属性只由 `attach_*` 的入参填）跳过并出版枚数，不许静默算过。
    """

    import ast
    import re

    probe = _load_probe()
    adapter_tree = ast.parse((PROJECT_ROOT / "taiji" / "adapter.py").read_text(encoding="utf-8"))
    slot_classes: dict[str, set[str]] = {}
    for node in ast.walk(adapter_tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                for cls in re.findall(r"([A-Z][A-Za-z0-9_]*)", ast.unparse(node.value)):
                    slot_classes.setdefault(target.attr, set()).add(cls)
    class_methods: dict[str, set[str]] = {}
    tracked = subprocess.run(
        ["git", "ls-files", "--", "taiji/*.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for rel in tracked:
        tree = ast.parse((PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    n.name
                    for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
                class_methods.setdefault(node.name, set()).update(methods)
    checked = 0
    skipped: list[str] = []
    bad: list[str] = []
    for label, path, method in probe.PARTS:
        if len(path) != 1:
            skipped.append(label)
            continue
        classes = {c for c in slot_classes.get(path[0], set()) if c != "None"}
        if not classes:
            #: 槽位只由 `attach_*` 的入参填，静态读不出类型——跳过并出版，不许静默算过。
            skipped.append(label)
            continue
        checked += 1
        if not any(method in class_methods.get(c, set()) for c in classes):
            bad.append(label)
    assert checked > 0, "一格都没核到＝这台自证是空验收"
    assert bad == [], f"这些格子的方法名不在被赋的类上：{bad}"
    assert len(skipped) < len(probe.PARTS), "全部格子都没被核到＝自证没有覆盖面"


def test_default_output_target_is_not_the_sealed_artifact() -> None:
    probe = _load_probe()
    parser = argparse.ArgumentParser()
    # 复用仪器自己的旗标定义：直接把 main() 里的 parser 造出来太重，这里按源码取默认值。
    src = SCRIPT.read_text(encoding="utf-8")
    assert (
        f'default="{SEALED}"' not in src
    ), f"--out-report 的默认值仍指向已封存件 {SEALED}：直接重跑会覆写历史读数"


def test_existing_tracked_target_is_refused_without_a_force_flag(tmp_path: Path) -> None:
    guard = getattr(_load_probe(), "_guard_overwrite", None)
    assert guard is not None, "仪器没有落盘护栏函数 _guard_overwrite"
    target = tmp_path / "already_there.json"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        guard(target, force=False)
    assert target.read_text(encoding="utf-8") == "{}\n"
    guard(target, force=True)


def test_probe_help_lists_both_new_columns() -> None:
    """旗标冒烟：证明帮助面真被走到，而不是只存在于源码里。"""

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "--overwrite" in proc.stdout
