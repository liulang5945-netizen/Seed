"""DEBT-I9 的两半：产品默认检查点的**可检测性**与**来源登记**。

背景：`checkpoints/*.pt` 不入 git，而应用启动时加载的就是"盘上那份"。2026-09-18 之前四次全量
套件每次都静默改写它（其中一次把 tick 从 36 退回 2）。写侧隔离已落地（DEBT-I7），但那条 fixture
只在 pytest 进程内生效——**测试起的子进程仍会落到真实默认路径**。所以本文件保证的不是"写不到"，
而是"写了会被发现"。

第二半是来源。v1 清单只能记 `provenance: unknown`，因为当时盘上那份的自述是
`trainer=api_seed_runtime`——它是一台测试套件的产物，不是任何训练的结果。2026-09-20 所有者裁决
把默认入口换到 16M-tick 训练态（`trainer=train_seed_corpus`，见
`plans/reference/M5_DEFAULT_SUBSTRATE_SWITCH_CONTRACT_20260920.md`），于是"来源"第一次成为可登记的
东西。两侧的诚实要求不对称，本文件两侧都钉：

* **不许把 unknown 洗成 known**：来源凭据只是"信封自述＋进度流＋历史命令记录三者吻合"，
  日志已不在盘上 ⇒ 清单必须同时带 `provenance_limits`，缺了就红。
* **不许停在 unknown**：换底之后仍记 unknown 就是没做完登记 ⇒ 也红。

已知边界：干净克隆上没有 `*.pt` ⇒ 相关测试 skip。那不是把门做成永远绿，它的意思是"这台机器上
没有可被改写的默认入口"，CI 亦如此；本仓库工作机上文件存在，守卫真实生效。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from api.seed_runtime import DEFAULT_CHECKPOINT as PRODUCT_DEFAULT

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
    """清单必须真的指着产品默认入口——否则下面的守卫可以被悄悄挪到别的文件上。

    这里**不重写常量**：期望值直接取 `api.seed_runtime.DEFAULT_CHECKPOINT`，两处任一改动而另一处
    没跟上都会红（v1 时代两边各写一份字面串，正是漂移的来源）。
    """

    relative = PRODUCT_DEFAULT.resolve().relative_to(REPO.resolve()).as_posix()
    assert manifest["path"] == relative
    assert manifest["format"] == "seed-product-default-checkpoint-provenance-v2"


def test_the_product_default_is_not_a_test_suite_artifact(manifest) -> None:
    """默认入口不得再服务套件重初始化的产物——那正是 DEBT-I9 的正题。"""

    assert manifest["envelope_trainer"] != "api_seed_runtime"
    assert manifest["envelope_trainer"] == "train_seed_corpus"
    assert int(manifest["envelope_tick"]) > 0


def test_the_default_checkpoint_still_matches_its_recorded_bytes(manifest) -> None:
    target = REPO / manifest["path"]
    if not target.is_file():
        pytest.skip(f"no {manifest['path']} on this machine (fresh clone: *.pt not in git)")

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


def test_the_envelope_claims_what_the_manifest_records(manifest) -> None:
    """清单里那几个字段必须真是信封里的值，不是照抄上一版。"""

    target = REPO / manifest["path"]
    if not target.is_file():
        pytest.skip(f"no {manifest['path']} on this machine")
    import torch

    blob = torch.load(target, map_location="cpu", weights_only=True)
    metadata = blob["metadata"]
    assert metadata["trainer"] == manifest["envelope_trainer"]
    assert int(metadata["tick"]) == int(manifest["envelope_tick"])
    assert metadata["saved_at_utc"] == manifest["envelope_saved_at_utc"]
    assert json.loads(metadata["corpus_fingerprint"])[0]["bytes"] == int(
        manifest["corpus_fingerprint"][0]["bytes"]
    )


def test_provenance_is_recorded_and_still_bounded(manifest) -> None:
    """来源既不许停在 unknown，也不许被写成比凭据更强的东西。"""

    assert manifest["provenance"] != "unknown"
    assert manifest["provenance_limits"], "登记了来源却不写边界 ⇒ 等于洗白"
    assert "DEBT-I9" in manifest["why_recorded"]
    assert manifest["before_L3_delivery"]


def test_the_switch_was_preflighted_on_the_bare_loader(manifest) -> None:
    """换底的前置问题（默认 loader 不加补丁能否加载这份旧格式训练态）必须有实测记录。

    若那一环将来退化成"靠 `relax_legacy_guard` 才打得开"，就不该把结论写成产品事实。
    """

    preflight = manifest["preflight_v0"]
    assert preflight["answer"] == "能"
    assert "冗余" in preflight["meaning"]
    assert preflight["contract"].startswith("plans/reference/")
    assert (REPO / preflight["contract"]).is_file()
