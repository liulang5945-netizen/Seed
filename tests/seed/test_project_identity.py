from __future__ import annotations

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
    assert 'name="Seed"' in spec.read_text(encoding="utf-8")
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
