from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 has no stdlib tomllib; tomli is its upstream implementation.
    import tomli as tomllib


REPO = Path(__file__).resolve().parents[2]


def test_distribution_and_readme_are_seed() -> None:
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (REPO / "README.md").read_text(encoding="utf-8")

    assert pyproject["project"]["name"] == "seed"
    assert "runtime" in pyproject["project"]["description"].lower()
    assert "seed*" in pyproject["tool"]["setuptools"]["packages"]["find"]["include"]
    assert readme.startswith("# Seed —")
    assert "Seed is the project" not in readme  # avoid reintroducing two identities


def test_desktop_build_artifact_is_seed() -> None:
    release = (REPO / "scripts" / "release.py").read_text(encoding="utf-8")
    installer = (REPO / "desktop" / "installer.nsi").read_text(encoding="utf-8")
    spec = REPO / "desktop" / "seed.spec"

    assert spec.is_file()
    assert not (REPO / "desktop" / "neuroplex.spec").exists()
    # 打包产物身份守护：唯一发布入口 release.py（原 build.py 已并入并删除，
    # 见 commit 52ee10c）现走 seed.spec 双入口（Seed.exe + SeedBackend.exe），
    # 主产物名必须是 Seed。
    assert "seed.spec" in release
    assert '"--clean"' in release
    assert 'name="Seed"' in spec.read_text(encoding="utf-8")
    assert "_append_data_tree" in spec.read_text(encoding="utf-8")
    assert '!define APP_EXE "Seed.exe"' in installer


def test_legacy_neuroplex_is_explicitly_a_frozen_comparison() -> None:
    direction = REPO / "plans" / "active" / "ARCHITECTURE_DIRECTION_2026_08.md"
    text = direction.read_text(encoding="utf-8")

    assert (REPO / "neuroplex").is_dir()
    assert "**Legacy NeuroPlex**" in text
    assert "冻结的 Transformer 基线" in text


def test_taiji_is_the_cognitive_architecture_and_seed_is_the_runtime() -> None:
    active = REPO / "plans" / "active"
    direction = (active / "ARCHITECTURE_DIRECTION_2026_08.md").read_text(encoding="utf-8")
    seed_architecture = (active / "SEED_ARCHITECTURE.md").read_text(encoding="utf-8")
    core_requirements = active / "TAIJI_CORE_REQUIREMENTS.md"
    taiji_architecture = active / "TAIJI_NATIVE_ARCHITECTURE_V1.md"
    archived_kernel = (
        REPO / "plans" / "archive" / "implementation" / "TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md"
    )

    assert core_requirements.is_file()
    assert taiji_architecture.is_file()
    assert archived_kernel.is_file()
    assert not (active / "TAIJI_SUBSTRATE_ARCHITECTURE.md").exists()
    assert {path.name for path in active.glob("*.md")} == {
        "ARCHITECTURE_DIRECTION_2026_08.md",
        "SEED_ARCHITECTURE.md",
        "SEED_DEVELOPMENT_ROADMAP_2026_08.md",
        "TAIJI_CORE_REQUIREMENTS.md",
        "TAIJI_NATIVE_ARCHITECTURE_V1.md",
    }
    assert "Taiji 是完整原生认知架构" in direction
    assert "Seed 是项目、产品和运行时" in direction
    assert "Seed 可以决定" in seed_architecture
    assert "不能决定" in seed_architecture


def test_active_plans_have_one_execution_owner_and_resolvable_links() -> None:
    active = REPO / "plans" / "active"
    roadmap = active / "roadmap"
    current = roadmap / "03_CURRENT_EXECUTION.md"

    # 契约：01~04 是 roadmap 骨干（必须存在）；05 起是随主线推进增生的登记类
    # 文档（技术债登记、决策记录等）。这里不穷举 05+ 的**具体文件名**，因为
    # 每新增一份登记文档都要改测试，会诱发「改测试而不是改事实」的坏习惯 ——
    # 与下方 execution_owner 断言当初放宽的动机相同。
    # 保留的约束是：骨干四件必须在位，且 01~09 编号下不得出现非 .md 杂物。
    backbone = {
        "01_SCOPE_AND_PHASES.md",
        "02_GATES_AND_CI.md",
        "03_CURRENT_EXECUTION.md",
        "04_EXECUTION_PLAN.md",
    }
    present = {path.name for path in roadmap.glob("*.md")}
    assert backbone <= present, f"roadmap backbone missing: {backbone - present}"
    assert all(
        path.is_file() and path.suffix == ".md" for path in roadmap.glob("0*")
    ), "roadmap shards must all be markdown files"

    execution_headings: list[tuple[Path, str]] = []
    for path in active.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        execution_headings.extend(
            (path, line)
            for line in text.splitlines()
            if line.startswith("## ") and ("当前唯一下一步" in line or "唯一执行项" in line)
        )

        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = match.group(1).split("#", 1)[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (
                (path.parent / target).resolve().exists()
            ), f"active plan link is missing: {path.relative_to(REPO)} -> {target}"

    # 唯一执行项断言只校验「唯一性 + 归属文件 + 固定前缀」，不把当前阶段名
    # 硬编码进测试。原因：该断言此前把 P5.2a 阶段名写死，导致每次 roadmap
    # 推进到下一阶段（P5.2b/P5.2c）都必须改测试，否则测试成为推进的阻力，
    # 并诱发「改测试而不是改事实」的坏习惯。契约保留的部分是：活跃计划树
    # 内只允许出现一条执行项声明，且必须位于 03_CURRENT_EXECUTION.md。
    assert (
        len(execution_headings) == 1
    ), f"expected exactly one execution owner, got {execution_headings}"
    owner_path, owner_heading = execution_headings[0]
    assert owner_path == current, f"execution owner must be {current}, got {owner_path}"
    assert owner_heading.startswith("## 当前唯一下一步：")
