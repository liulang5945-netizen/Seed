"""DEBT-I7（第二例的根治）：测试不得写产品正在使用的工件目录。

产品也会服务 ``output/manual`` + ``-r5-canary/`` 这个目录。测试曾经在它里面按
``sNN-<kind>-<pid>`` 建根目录，每个检查点 ~43 MB；跑到 teardown 的会话能自我清理，**被终止的
进程不能** —— 2026-09-19 实测该目录积了 12 个文件 / 496 MiB，全部来自 6 个已经死掉的 pid。
会话末 sweep 只是兜底，不是修法。

现在的修法是让测试根本不拼那个路径（改道见 ``tests/_scratch.py``），本文件钉住三件事：
① 静态扫源码：``tests/`` 下除 ``_scratch.py``（唯一被许可的读方）外，不许再出现该路径拼接；
② 改道方向本身要钉住：scratch 根必须在仓库外，否则 ① 形同虚设；
③ 反向不误伤：那个目录是产品的资产目录（``README.md`` + ``native-canary.pt`` 是版本化验收证据），
   守卫只约束 ``tests/``，不约束生产代码，也不许被顺手 ignore 掉。
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: 相邻两段字面量才算"拼出这个路径"；散文里提到目录名不算。
_PRODUCT_STORE_PATH = re.compile(r'"output"\s*/\s*"manual[^"]*canary"')

#: 唯一被许可命名该目录的文件（只读，用于第 ③ 条断言）。
SANCTIONED_READER = "_scratch.py"


def test_no_test_code_paths_into_the_products_artifact_store() -> None:
    """静态扫源码而不是扫磁盘：磁盘上的残留会被清掉，拼路径那行代码却会一直生产残留。"""

    offenders: list[str] = []
    for path in sorted((REPO / "tests").rglob("*.py")):
        if path.name == SANCTIONED_READER:
            continue
        if _PRODUCT_STORE_PATH.search(path.read_text(encoding="utf-8")):
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], f"测试代码仍在拼产品工件库路径（应改用 tests/_scratch.py）：{offenders}"


def test_scratch_root_lives_outside_the_repository() -> None:
    """改道的方向也得钉住：scratch 根若落在仓库内，第 ① 条就形同虚设。"""

    from _scratch import artifact_scratch_root

    root = artifact_scratch_root()
    assert not root.is_relative_to(REPO), f"scratch 根仍在仓库内：{root}"
    assert root.is_dir()


def test_product_store_dir_holds_only_versioned_assets() -> None:
    """产品那个目录只应有版本化资产；测试跑完不该往里添加任何东西。"""

    from _scratch import PRODUCT_STORE_DIR

    if not PRODUCT_STORE_DIR.is_dir():
        return
    extra = sorted(
        entry.name
        for entry in PRODUCT_STORE_DIR.iterdir()
        if entry.name not in {"README.md", "native-canary.pt"}
    )
    assert extra == [], f"产品工件目录里出现测试残留（测试应写在临时目录）：{extra}"


def test_the_sanctioned_reader_is_the_only_naming_site() -> None:
    """豁免面必须是**一个具名文件**，而不是"看起来不像路径"的模糊规则。"""

    named = [
        path.name
        for path in sorted((REPO / "tests").rglob("*.py"))
        if "PRODUCT_STORE_DIR" in path.read_text(encoding="utf-8")
    ]
    assert named == [SANCTIONED_READER, "test_artifact_store_scratch_contract.py"], named
