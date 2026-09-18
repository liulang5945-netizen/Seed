"""DEBT-I9 的可检测半：产品默认检查点一旦被改写，就要有测试红。

背景：`checkpoints/seed_corpus.pt` 是应用启动时加载的那份模型，而 ``*.pt`` 不入 git。
2026-09-18 之前，四次全量套件**每次都静默改写它**（其中一次把 tick 从 36 退回 2）。
写侧隔离已落地（DEBT-I7：`DEFAULT_SAVE_TARGET` + 会话级重定向），但那条 fixture 只在 pytest
进程内生效——**测试起的子进程仍然会落到真实默认路径**。所以本文件保证的不是"写不到"，
而是"写了会被发现"，这是那四次之所以无人知晓的直接补救。

已知边界（不粉饰）：
* 这只解决**可检测性**。DEBT-I9 的正题——默认基座的**来源**——仍未解决：现在盘上这份是
  "某次套件重初始化"的产物（envelope 自己写着 ``trainer=api_seed_runtime`` 与套件结束那一刻的
  ``saved_at_utc``），不是出厂基座。要结项需要一份非测试产生的基座或官方重训 + 重采。
* 干净克隆上没有 ``*.pt`` ⇒ 第一条测试会 skip。那不是把门做成永远绿：它的意思是"这台机器上
  没有可被改写的默认入口"，CI 亦如此；本仓库的工作机上该文件存在，守卫真实生效。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "plans" / "manifests" / "product_default_checkpoint_provenance.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def test_the_recorded_path_is_the_product_default(manifest) -> None:
    """清单必须真的指着产品默认入口——否则下面的守卫可以被悄悄挪到别的文件上。"""

    assert manifest["path"] == "checkpoints/seed_corpus.pt"
    assert manifest["format"] == "seed-product-default-checkpoint-provenance-v1"


def test_the_default_checkpoint_still_matches_its_recorded_bytes(manifest) -> None:
    target = REPO / manifest["path"]
    if not target.is_file():
        pytest.skip("no checkpoints/seed_corpus.pt on this machine (fresh clone: *.pt not in git)")

    recorded = manifest["sha256"]
    actual = _sha256(target)
    assert actual == recorded, (
        f"产品默认检查点被改写：现 sha {actual[:12]}…，在册 {recorded[:12]}…。\n"
        "两种可能都必须查清，不许直接把期望值改成当前值：\n"
        "  ① 某条测试（或它起的子进程）又写了默认路径 ⇒ DEBT-I7 的漏口，先修写者；"
        "conftest 的重定向只覆盖 pytest 进程内，不覆盖子进程。\n"
        "  ② 有人有意重训/替换了基座 ⇒ 更新本清单，并在 05 债册 DEBT-I9 记一笔新来源。"
    )
    assert target.stat().st_size == manifest["bytes"]


def test_the_manifest_admits_that_its_own_provenance_is_unknown(manifest) -> None:
    """这份记录不许被"记下来"洗成"官方的"：来源必须仍写明 unknown，且承认是套件产物。"""

    assert manifest["provenance"] == "unknown"
    assert manifest["envelope_trainer"] == "api_seed_runtime"
    assert manifest["envelope_tick"] == 2
    assert "DEBT-I9" in manifest["why_recorded"]
    assert manifest["before_L3_delivery"]
